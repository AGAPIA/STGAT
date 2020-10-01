import argparse
import os
import sys
sys.path.append(os.getcwd())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import config

from data.loader import data_loader
from models import TrajectoryGenerator
from utils import (
    displacement_error,
    final_displacement_error,
    l2_loss,
    int_tuple,
    relative_to_abs,
    get_dset_path,
)

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")

parser.add_argument("--dataset_name", default="zara2", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=4, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=8, type=int)
parser.add_argument("--skip", default=1, type=int)
parser.add_argument('--external_test', type=int, default=0)
parser.add_argument('--external', type=int, default=1)
parser.add_argument('--use_gpu', default=0, type=int)

parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--batch_size", default=64, type=int)

parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")
parser.add_argument("--noise_mix_type", default="global")

parser.add_argument(
    "--traj_lstm_input_size", type=int, default=2, help="traj_lstm_input_size"
)
parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)

parser.add_argument(
    "--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma"
)
parser.add_argument(
    "--hidden-units",
    type=str,
    default="16",
    help="Hidden units in each hidden layer, splitted with comma",
)
parser.add_argument(
    "--graph_network_out_dims",
    type=int,
    default=32,
    help="dims of every node after through GAT module",
)
parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)


parser.add_argument("--num_samples", default=20, type=int)


parser.add_argument(
    "--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)

parser.add_argument("--dset_type", default="test", type=str)


parser.add_argument(
    "--model_path",
    default="./model_best.pth.tar",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def get_generator(checkpoint):
    n_units = (
        [args.traj_lstm_hidden_size]
        + [int(x) for x in args.hidden_units.strip().split(",")]
        + [args.graph_lstm_hidden_size]
    )
    n_heads = [int(x) for x in args.heads.strip().split(",")]
    model = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        traj_lstm_input_size=args.traj_lstm_input_size,
        traj_lstm_hidden_size=args.traj_lstm_hidden_size,
        n_units=n_units,
        n_heads=n_heads,
        graph_network_out_dims=args.graph_network_out_dims,
        dropout=args.dropout,
        alpha=args.alpha,
        graph_lstm_hidden_size=args.graph_lstm_hidden_size,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(config.g_device)
    model.eval()
    return model


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt, mode="raw")
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode="raw")
    return ade, fde


def evaluate(args, loader, generator):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.to(config.g_device) for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(args.num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj_rel, obs_traj, seq_start_end, 0, 3
                )
                pred_traj_fake_rel = pred_traj_fake_rel[-args.pred_len :]

                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
                ade.append(ade_)
                fde.append(fde_)
            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde


def main(args):
    checkpoint = torch.load(args.resume)
    generator = get_generator(checkpoint)
    path = get_dset_path(args.dataset_name, args.dset_type)

    _, loader = data_loader(args, path)
    ade, fde = evaluate(args, loader, generator)
    print(
        "Dataset: {}, Pred Len: {}, ADE: {:.12f}, FDE: {:.12f}".format(
            args.dataset_name, args.pred_len, ade, fde
        )
    )

# FLASK SESSION GLOBAL DEFINES
import numpy as np
from flask import Flask, jsonify, request
import json
import io
app = Flask(__name__)

NUM_FRAMES_TO_OBSERVE = 8
NUM_FRAMES_TO_PREDICT = 8
MAIN_AGENT_NAME = "main" # The agent under evaluation
MAIN_AGENT_INDEX = 0
generator = None
history_pos = {} # History of positons for a given agent name
#----------

# This code runs the inference for one frome
# Input params:
# agentsObservedPos dict of ['agentName'] -> position as np array [2], all agents observed in this frame
# optional: forcedHistoryDict -> same as above but with NUM_FRAMES_OBSERVED o neach agent, allows you to force / set history
# Output : returns the position of the 'main' agent
def DoModelInferenceForFrame(agentsObservedOnFrame, forcedHistoryDict = None):
    global history_pos

    # Update the history if forced param is used
    if forcedHistoryDict != None:
        for key, value in forcedHistoryDict.items():
            assert isinstance(value, np.ndarray), "value is not instance of numpy array"
            assert value.shape is not (NUM_FRAMES_TO_OBSERVE, 2)
            history_pos[key] = value

    # Update the history of agents seen with the new observed values
    for key, value in agentsObservedOnFrame.items():
        # If agent was not already in the history pos, init everything with local value
        if key not in history_pos:
            history_pos[key] = np.tile(value, [NUM_FRAMES_TO_OBSERVE, 1])
        else:  # Else, just put his new pos in the end of history
            values = history_pos[key]
            values[0:NUM_FRAMES_TO_OBSERVE - 1] = values[1:NUM_FRAMES_TO_OBSERVE]
            values[NUM_FRAMES_TO_OBSERVE - 1] = value

    # Do simulation using the model
    # ------------------------------------------

    # Step 1: fill the input
    numAgentsThisFrame = len(agentsObservedOnFrame)

    # Absolute observed trajectories
    obs_traj = np.zeros(shape=(NUM_FRAMES_TO_OBSERVE, numAgentsThisFrame, 2), dtype=np.float32)
    # Zero index is main, others are following
    obs_traj[:, MAIN_AGENT_INDEX, :] = history_pos[MAIN_AGENT_NAME]
    index = 1
    indexToAgentNameMapping = {}
    indexToAgentNameMapping[MAIN_AGENT_INDEX] = MAIN_AGENT_NAME
    for key, value in agentsObservedOnFrame.items():
        if key != MAIN_AGENT_NAME:
            obs_traj[:,index,:] = history_pos[key]
            indexToAgentNameMapping[index] = key
            index += 1

    # Relative observed trajectories
    obs_traj_rel = np.zeros(shape=(NUM_FRAMES_TO_OBSERVE, numAgentsThisFrame, 2), dtype=np.float32)
    obs_traj_rel[1:, :, :] = obs_traj[1:, :, :] - obs_traj[:-1, :, :]

    seq_start_end = np.array([[0, numAgentsThisFrame]])  # We have only 1 batch containing all agents
    # Transform them to torch tensors
    obs_traj = torch.from_numpy(obs_traj).type(torch.float)
    obs_traj_rel = torch.from_numpy(obs_traj_rel).type(torch.float)
    seq_start_end = torch.from_numpy(seq_start_end)

    pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    # Take the first predicted position and add it to history
    pred_traj_fake = pred_traj_fake.detach().numpy()
    newMainAgentPos = pred_traj_fake[0][0]  # Agent 0 is our main agent

    return newMainAgentPos


@app.route('/predict', methods=['POST'])
def predict():
    if request.method =='POST':
        # Get the file from request
        #file = request.files['file']
        dataReceived = json.loads(request.data)

        agentsObservedThisFrame = dataReceived['agentsPosThisFrame']
        agentsForcedHistory = dataReceived['agentsForcedHistoryPos'] if 'agentsForcedHistoryPos' in dataReceived else None

        # Read all agents data received
        #-----------------------------------------
        #agentIndex = 1
        agentsObservedPos = {}
        for key,value in agentsObservedThisFrame.items():
            value = np.array(value, dtype=np.float32)
            if key == MAIN_AGENT_NAME:
                agentsObservedPos[MAIN_AGENT_NAME] = value
            else:
                agentsObservedPos[key] = value

        forcedHistoryPos = None
        if agentsForcedHistory is not None:
            forcedHistoryPos = {}
            for key, value in agentsForcedHistory.items():
                value = np.array(value, dtype=np.float32)
                if key == MAIN_AGENT_NAME:
                    forcedHistoryPos[MAIN_AGENT_NAME] = value
                else:
                    forcedHistoryPos[key] = value

        # Then do model inference for agents observed on this frame
        # Get back the new position for main agent and return it to caller
        newMainAgentPos = DoModelInferenceForFrame(agentsObservedPos, forcedHistoryPos)
        return jsonify(newMainAgentPos = str(list(newMainAgentPos)))


def deloyModelForFlaskInference():
    checkpoint = torch.load(args.model_path)

    global generator
    generator = get_generator(checkpoint)
    #_args = AttrDict(checkpoint['args'])
    #_args['loader_num_workers'] = 0

def startExternalTest():
    deloyModelForFlaskInference()

    historyMainAgent = np.zeros(shape=(NUM_FRAMES_TO_OBSERVE, 2))
    agentsObservedPos = {MAIN_AGENT_NAME : np.array([0,0], dtype=np.float32)}

    for frameIndex in range(100):
        forcedHistory = None
        if frameIndex == 0:
            forcedHistory = {MAIN_AGENT_NAME : historyMainAgent}

        newMainAgentPos = DoModelInferenceForFrame(agentsObservedPos, forcedHistory)
        print(f"Frame {frameIndex}: {newMainAgentPos}")
        agentsObservedPos[MAIN_AGENT_NAME] = newMainAgentPos

"""
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
     non_linear_ped, loss_mask, seq_start_end) = batch

    ade, fde = [], []
    total_traj += pred_traj_gt.size(1)

    for _ in range(num_samples):
        pred_traj_fake_rel = generator(
            obs_traj, obs_traj_rel, seq_start_end
        )
        pred_traj_fake = relative_to_abs(
            pred_traj_fake_rel, obs_traj[-1]
        )
        ade.append(displacement_error(
            pred_traj_fake, pred_traj_gt, mode='raw'
        ))
        fde.append(final_displacement_error(
            pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
        ))
"""

if __name__ == "__main__":
    args = parser.parse_args()
    args.resume = args.model_path
    args.model_path = args.resume
    config.initDevice(args.use_gpu)

    torch.manual_seed(72)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if True and args.external == 1:
        deloyModelForFlaskInference()

        if "carla" in args.resume:
            print("Running deployment for Carla...")
            app.run(host="localhost", port=5200, debug=False)
        elif "waymo" in args.resume:
            print("Running deployment for Waymo...")
            app.run(host="localhost", port=5201, debug=False)
        else:
            print("Running deployment for others...")
            app.run(host="localhost", port=8200, debug=False)
        app.run()
    elif True and args.external_test == 1:
        startExternalTest()
    else: # normal evaluation
        main(args)
