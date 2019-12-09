import numpy as np
import logging
from tqdm import tqdm
from utils.config import *
from utils.utils_sensation_lcsts import *
from torch.nn.utils import clip_grad_norm
from seq2seq.sensation_get_to_the_point import *
from seq2seq.sensation_scorer import SensationCNN
import logging
import copy
import jieba
from utils.function import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer(object):
    def __init__(self):

        args = NNParams().args
        logging.info(args["thd"])
        train, dev, test, lang, max_q, max_r = prepare_data_seq(batch_size=args['batch_size'], debug=args["debug"], shuffle=True, pointer_gen=args["pointer_gen"], output_vocab_size=args['output_vocab_size'], thd=args["thd"])
        args["vocab_size"] = lang.n_words
        args["max_q"] = max_q
        args["max_r"] = max_r

        self.args = args
        self.train = train
        self.dev = dev
        self.test = test
        self.lang = lang

        # model = globals()[args["model_type"]](args, lang, max_q, max_r)
        model = PointerAttnSeqToSeq(self.args, lang)
        self.model = model
        if USE_CUDA:
            self.model = self.model.cuda()

        logging.info(model)
        logging.info("encoder parameters: {}".format(count_parameters(model.encoder)))
        logging.info("decoder parameters: {}".format(count_parameters(model.decoder)))
        logging.info("embedding parameters: {}".format(count_parameters(model.embedding)))
        logging.info("model parameters: {}".format(count_parameters(model)))

        self.loss, self.acc, self.reward, self.print_every = 0.0, 0.0, 0.0, 1

        assert args["sensation_scorer_path"] is not None
        opts = torch.load(args["sensation_scorer_path"]+"/args.th")
        self.sensation_model = SensationCNN(opts, self.lang)
        logging.info("load checkpoint from {}".format(args["sensation_scorer_path"]))
        checkpoint = torch.load(args["sensation_scorer_path"]+"/sensation_scorer.th")
        self.sensation_model.load_state_dict(checkpoint['model'])
        if USE_CUDA:
            self.sensation_model.cuda()

        if self.args['optimizer'] == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'])
        elif self.args['optimizer'] == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args['lr'])
        else:
            raise ValueError("optimizer not implemented")

    def save_model(self, save_name, best_result, step):
        directory = "sensation_save/" + save_name + "/"
        directory = directory + "_".join([str(self.args[a]) for a in save_params]) + "_" + str(best_result)
        if not os.path.exists(directory):
            os.makedirs(directory)
        ckpt = {"model": self.model.state_dict(),  "step": step, "optimizer": self.optimizer.state_dict(),
         "best_result":best_result}
        torch.save(self.args, directory+"/args.th")
        if self.args["use_rl"]:
            ckpt["rl_optimizer"] = self.rl_optimizer
            torch.save(ckpt, directory+"/rl.th")
        else:
            torch.save(ckpt, directory+"/get_to_the_point.th")
        return directory
 
    def load_base_model(self):
        path = self.args["path"]
        ckpt = torch.load(path+"/get_to_the_point.th")
        logging.info("load ckpt from {}, step is {}, best_result {}".format(path, ckpt["step"], ckpt["best_result"]))

        self.model.load_state_dict(ckpt["model"])
        if not self.args["use_rl"]:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt["step"], ckpt["best_result"]

    def load_rl_model(self):
        path = self.args["rl_model_path"]
        ckpt = torch.load(path+"/rl.th")
        logging.info("load ckpt from {}, step is {}, best_result {}".format(path, ckpt["step"], ckpt["best_result"]))

        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.rl_optimizer = ckpt["rl_optimizer"]
        return ckpt["step"], ckpt["best_result"]

    def print_loss(self, step):
        print_loss_avg = self.loss / self.print_every
        print_acc_avg = self.acc / self.print_every
        print_reward_avg = self.reward / self.print_every
        if self.args["use_rl"]:
            print_expected_rewards_loss_avg = self.expected_rewards_loss / self.print_every
        self.print_every += 1
        if self.args["use_rl"]:
            return 'step: {}, L:{:.2f}, acc:{:.2f}, r:{:.3f}, r_loss:{:.4f}'.format(step, print_loss_avg, print_acc_avg, print_reward_avg, print_expected_rewards_loss_avg)
        else:
            return 'step: {}, L:{:.2f}, acc:{:.2f}, r:{:.3f}'.format(step, print_loss_avg, print_acc_avg, print_reward_avg)

    def train_step(self, batch, step, reset):

        if reset:
            self.loss = 0.0
            self.acc = 0.0
            self.reward = 0.0
            self.print_every = 1
            if self.args["use_rl"]:
                self.expected_rewards_loss = 0.0

        self.optimizer.zero_grad()
        assert self.args["use_s_score"] is not None
        if self.args["use_rl"]:
            r, loss, acc, expected_rewards_loss, _ = self.model.get_rl_loss(batch, self.sensation_model, use_s_score=self.args["use_s_score"])
        else:
            _, loss, acc = self.model.get_loss(batch)

        loss.backward()

        clip_grad_norm(self.model.parameters(), self.args["max_grad_norm"])

        self.optimizer.step()

        self.loss += loss.data[0] 
        self.acc += acc.data[0]
        if self.args["use_rl"]:
            self.reward += r.data[0]

        if self.args["use_rl"]:
            self.rl_optimizer.zero_grad()
            expected_rewards_loss.backward()
            self.rl_optimizer.step()
            self.expected_rewards_loss += expected_rewards_loss.data[0]

    def d_step(self, batch, probs, backward=True):

        ## from seq_first to batch first
        input_batch = batch["input_batch"].transpose(0,1)
        target_batch = batch["target_batch"].transpose(0,1)
        batch_size = input_batch.size(0)

        true_prob = self.D(input_batch, target_batch)
        true_labels = Variable(torch.ones(batch_size))
        if USE_CUDA:
            true_labels = true_labels.cuda()
        true_loss = F.binary_cross_entropy(true_prob, true_labels)
        if backward:
            true_loss.backward()
        true_acc = ((true_prob > 0.5).long() == true_labels.long()).float().sum() * 1.0 / true_labels.size(0)

        fake_labels = Variable(torch.zeros(batch_size))
        if USE_CUDA:
            fake_labels = fake_labels.cuda()
        fake_loss = F.binary_cross_entropy(probs, fake_labels)
        if backward:
            fake_loss.backward(retain_graph=True)
        fake_acc = ((probs > 0.5).long() == fake_labels.long()).float().sum() * 1.0 / fake_labels.size(0)

        loss = (true_loss.data[0] + fake_loss.data[0]) / 2
        # acc = (true_acc +  fake_acc) / 2
        if backward:
            logging.info("true loss is  {}, fake  loss is {}, true acc is {}, false acc is {}".format(true_loss, fake_loss, true_acc, fake_acc))
        return true_acc.data[0], fake_acc.data[0]

    def training(self):

        # Configure models
        step = 0
        best_metric = 0.0
        cnt = 0

        if self.args["use_rl"] and self.args["path"] is None and self.args["rl_model_path"] is None:
            raise ValueError("use rl but path is not given")

        if self.args["use_rl"] is None and self.args["rl_model_path"] is not None:
            raise ValueError("not using rl but give rl_model_path")

        if self.args["rl_model_path"] is not None:
            self.model.expected_reward_layer = torch.nn.Linear(self.args["hidden_size"], 1)
            if USE_CUDA:
                self.model.expected_reward_layer = self.model.expected_reward_layer.cuda()
            self.rl_optimizer = torch.optim.Adam(self.model.expected_reward_layer.parameters(), lr=self.args["rl_lr"])
            step, best_metric = self.load_rl_model()
        elif self.args["path"] is not None:
            step, _ = self.load_base_model() 
            if self.args["use_rl"]:
                best_metric = 0.0
                self.model.expected_reward_layer = torch.nn.Linear(self.args["hidden_size"], 1)
                if USE_CUDA:
                    self.model.expected_reward_layer = self.model.expected_reward_layer.cuda()
                self.rl_optimizer = torch.optim.Adam(self.model.expected_reward_layer.parameters(), lr=self.args["rl_lr"])
        else:
            pass
        self.old_model = copy.deepcopy(self.model) 
        total_steps = self.args["total_steps"]
        while step < total_steps:
            for j, batch in enumerate(self.train):
                if self.args['debug'] and j>1100:
                    break
                
                if not self.args["debug"]:
                    logging_step = 1000
                else:
                    logging_step = 10

                if j % logging_step == 0:
                    # if self.args["use_rl"]:
                    #     save_folder = "logs/Rl/"+"_".join([str(self.args[a]) for a in save_params]) 
                    #     os.makedirs(save_folder, exist_ok=True)
                    #     self.save_decode_sents(self.test, save_folder+"/prediction_step_{}.txt".format(step))
                    hyp, ref = self.model.predict_batch(batch, self.args["decode_type"])
                    old_hyp, _ = self.old_model.predict_batch(batch, self.args["decode_type"])
                    decoded_sents = self.model.decode_batch(batch,"beam")

                    sensation_rewards = self.model.get_sensation_reward(decoded_sents, batch,  self.sensation_model)
                    rewards = self.model.get_reward(decoded_sents, batch, self.sensation_model)[0]
                    for i,(prediction, ground_truth, old_pred) in enumerate(zip(hyp, ref, old_hyp)):
                        logging.info("prediction: {}".format(prediction))
                        logging.info("seq2seq prediction: {}".format(old_pred))
                        logging.info("prediction sensation score: {}, {}".format(sensation_rewards[i], rewards[i]))
                        if self.args["use_rl"]:
                            rouge_rewards = self.model.compute_rouge_reward(list(jieba.cut("".join(prediction.split()))), batch["input_txt"][i], batch["target_txt"][i])
                            logging.info("rouge_r: {},  lexicon_r: {}, arousal_r:{}, reward:{}".format(rouge_rewards, sensation_rewards[i], 0.0, rewards[i]))
                        logging.info("ground truth: {}".format(ground_truth))
                        logging.info("ground sensation score: {}".format(batch["sensation_scores"][i]))
                        logging.info("input article: {}".format(batch["input_txt"][i]))
                        logging.info("decode type: {}, {}: {}".format(self.args["decode_type"], rouge_metric, rouge([prediction], [ground_truth])[rouge_metric]))

                if step % int(self.args['eval_step']) == 0: 
                    dev_metric, _, (hyp, ref, rewards, sensation_scores, articles) = self.model.evaluate(self.dev, self.args["decode_type"], sensation_model=self.sensation_model, return_pred=True)
                    if(dev_metric > best_metric):
                        best_metric = dev_metric
                        cnt=0
                        if self.args["use_rl"]:
                            directory = self.save_model("Rl", best_metric, step)
                            with open(directory + "/prediction", "w") as f:
                                f.write("\n".join(["{}\t{:.5f}\n{}\t{:.5f}\n{}\n".format(h,r,g,s,a) for h,g,r,s,a in zip(hyp, ref, rewards, sensation_scores, articles)]))
                        else:
                            directory = self.save_model("PointerAttn_Scratch", best_metric, step)
                            with open(directory + "/prediction", "w") as f:
                                f.write("\n".join(["{}\t{:.5f}\n{}\t{:.5f}\n{}\n".format(h,r,g,s,a) for h,g,r,s,a in zip(hyp, ref, rewards, sensation_scores, articles)]))
                    else:
                        cnt+=1
                    if(cnt == 5): 
                        ## early stopping
                        step = total_steps + 1
                        break

                self.train_step(batch, step, j==0)
                logging.info(self.print_loss(step))
                step += 1


    def save_decode_sents(self, data, save_file):

        logging.info("start decoding")
        hyp = []
        ref = []
        article = []
        # pbar = tqdm(enumerate(dev), total=len(dev))
        # for j, data_dev in pbar:
        rewards = []
        rouge_r = []
        sensation_rewards = []
        for j, data_dev in enumerate(data):

            decoded_sents = self.model.decode_batch(data_dev, "beam")
            if self.args["use_rl"]:
                sensation_rewards.extend([r for r in self.model.get_sensation_reward(decoded_sents, data_dev, self.sensation_model)])
                rewards.extend([ r for r in self.model.get_reward(decoded_sents, data_dev, self.sensation_model)[0] ])
            for i, sent in enumerate(decoded_sents):
                hyp.append(" ".join("".join(sent)))
                ref.append(" ".join("".join(data_dev["target_txt"][i].split())))
                article.append(data_dev["input_txt"][i])
                if self.args["use_rl"]:
                    rouge_r.append(self.model.compute_rouge_reward(sent, data_dev["input_txt"][i], data_dev["target_txt"][i]))

        rouge_score = rouge(hyp, ref)
        with open(save_file, "w") as f:
            if self.args["use_rl"]:
                f.write("\n".join(["{}\nrouge_r: {},lexicon_r:{}, reward:{}\n{}\n{}\n".format(h,r_r,l_r,r,g,a) for h,g,r_r,l_r,r,a in zip(hyp, ref, rouge_r,sensation_rewards, rewards, article)]))
            else:
                f.write("\n".join([h+"\n"+g+"\n" for h,g in zip(hyp, ref)]))
            f.write("\n" + str(rouge_score) + "\n")
            f.write("rewards: " + str(sum(rewards) / len(rewards)) + "\n")

    def decoding(self, decode_type="beam"):
        # Configure models

        if self.args["use_rl"]  and self.args["rl_model_path"] is None:
            raise ValueError("use rl but path is not given")

        if self.args["use_rl"] is None and self.args["rl_model_path"] is not None:
            raise ValueError("not using rl but give rl_model_path")

        if self.args["rl_model_path"] is not None:
            self.model.expected_reward_layer = torch.nn.Linear(self.args["hidden_size"], 1)
            if USE_CUDA:
                self.model.expected_reward_layer = self.model.expected_reward_layer.cuda()
            self.rl_optimizer = torch.optim.Adam(self.model.expected_reward_layer.parameters(), lr=self.args["rl_lr"])
            step, _ = self.load_rl_model()
            save_file = self.args["rl_model_path"] + "/prediction.txt"
        elif self.args["path"] is not None:
            step, _ = self.load_base_model()
            if self.args["use_rl"]:
                self.model.expected_reward_layer = torch.nn.Linear(self.args["hidden_size"], 1)
                if USE_CUDA:
                    self.model.expected_reward_layer = self.model.expected_reward_layer.cuda()
                self.rl_optimizer = torch.optim.Adam(self.model.expected_reward_layer.parameters(), lr=self.args["rl_lr"])
            save_file = self.args["path"] + "/prediction.txt"
        else:
            pass

        _, _, (hyp, ref, rewards, sensation_scores, articles) = self.model.evaluate(self.test, self.args["decode_type"], sensation_model=self.sensation_model, return_pred=True)
        if self.args["rl_model_path"] is not None:
            directory = self.args["rl_model_path"]
            with open(directory + "/test_prediction", "w") as f:
                f.write("\n".join(["{}\t{:.5f}\n{}\t{:.5f}\n{}\n".format(h,r,g,s,a) for h,g,r,s,a in zip(hyp, ref, rewards, sensation_scores, articles)]))
                f.write("\n{}\n".format(str(rouge(hyp, ref))))
                
        elif self.args["path"] is not None:
            directory = self.args["path"]
            with open(directory + "/test_prediction", "w") as f:
                f.write("\n".join(["{}\t{:.5f}\n{}\t{:.5f}\n{}\n".format(h,r,g,s,a) for h,g,r,s,a in zip(hyp, ref, rewards, sensation_scores, articles)]))
                f.write("\n{}\n".format(str(rouge(hyp, ref))))

if __name__ == "__main__":
    trainer =  Trainer()
    trainer.training()
