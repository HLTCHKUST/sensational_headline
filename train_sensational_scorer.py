import numpy as np
import logging
from tqdm import tqdm

from utils.sensation_config import *
from utils.utils_sensation import *

from seq2seq.sensation_scorer import *
import logging
import os

# Configure models
best_metric = 0.0
cnt = 0
loss_steps = 0
loss = 0.0
accuracy = 0.0

args = CNNParams().args
train, dev, test, lang, max_q, max_r = prepare_data_seq(batch_size=args['batch_size'], debug=args["debug"], shuffle=True, output_vocab_size=args['output_vocab_size'])
args["vocab_size"] = lang.n_words


D = SensationCNN(args, lang)
optimizer = torch.optim.Adam(D.parameters(), 0.001)

logging.info(D)
logging.info("per epoch: {} steps".format(len(train)))
step = 0
total_steps = args["total_steps"]

if args["sensation_scorer_path"] is not None:
    logging.info("load checkpoint from {}".format(args["sensation_scorer_path"]))
    checkpoint = torch.load(args["sensation_scorer_path"]+"/sensation_scorer.th")
    step = checkpoint["step"]
    D.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint["optimizer"])

def print_loss():
    return "step {}, loss {:.4f}, acc {:.4f}".format(step, loss / loss_steps, accuracy / loss_steps)

num_error = 0
while step < total_steps:
    pbar = tqdm(enumerate(train), total=len(train))
    for i, batch in pbar:
        # logging.info("step:{}".format(step))  
        if args['debug'] and i>1100:
            break
        optimizer.zero_grad()
        l, acc = D.train_step(batch)
        optimizer.step()
        loss_steps += 1
        loss += l 
        accuracy += acc
        step += 1
        pbar.set_description(print_loss())

        if args["debug"] and step == 10:
            dev_metric = D.evaluate(dev)
            if(dev_metric > best_metric):
                best_metric = dev_metric
                cnt=0
                save_path = "save/sensation/"+"_".join([str(args[a]) for a in ["num_filters", "filter_sizes"]])+"_"+str(best_metric) 
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save({"step": step, "optimizer":optimizer.state_dict(), "model":D.state_dict()}, save_path + "/sensation_scorer.th")
                logging.info("save to {}".format(save_path))
  
    if True:
        dev_metric = D.evaluate(dev)
        logging.info("dev accuracy is  {}".format(dev_metric))
        if(dev_metric > best_metric):
            best_metric = dev_metric
            cnt=0
            save_path = "save/sensation/"+"_".join([str(args[a]) for a in ["num_filters"]])+"_"+str(float(best_metric) )
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save({"step": step, "optimizer":optimizer.state_dict(), "model":D.state_dict()}, save_path + "/sensation_scorer.th")
            torch.save(args, save_path + "/args.th")
            logging.info("save to {}".format(save_path))
        else:
            cnt+=1
        if(cnt == 5): 
            ## early stopping
            step = total_steps + 1
            break

    if step > total_steps:
        break
