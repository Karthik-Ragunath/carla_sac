# from customModel import CNNClassifier, save_model, load_model
from torch_base.torch_agent_inference_ppo import ImitationLearningAgent
from imitation_utils import accuracy, load_data
import torch
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
import os
import logging
from env_config import EnvConfig
import argparse

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')

parser.add_argument('-l', '--lrate', type=float, default=0.001)
parser.add_argument('-e', '--epochs', type=int, default=450)
parser.add_argument('-t', '--tbdir', type=str, default="")
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
# parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
# parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
# parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
# parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument("--device_id", "-dev", type=int, default=0, required=False)
parser.add_argument("--log_seed", type=str, default=0, required=False)
# parser.add_argument("--num_episodes", type=int, default=1, required=False)
parser.add_argument("--context", type=str, default='imitation_0', required=False)
# parser.add_argument("--num_steps_per_episode", type=int, default=250, required=False)
parser.add_argument("--load_context", type=str, required=False)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{args.device_id}" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

LOGGER= logging.getLogger()
LOGGER.setLevel(logging.DEBUG) # or whatever
handler = logging.FileHandler(f"ppo_logger_imitation_{args.log_seed}.log", 'w', 'utf-8')
formatter = logging.Formatter('%(name)s %(message)s')
handler.setFormatter(formatter)
LOGGER.addHandler(handler)

if __name__ == "__main__":
    """Main function."""
    checkpoints_save_dir = 'params_' + args.context
    if not os.path.exists(checkpoints_save_dir):
        os.makedirs(checkpoints_save_dir)

    agent = ImitationLearningAgent(device=device, env_params=EnvConfig['imitation_env_params'], context=args.context, args=args)
    
    dirname = os.path.join("runs", args.tbdir)
    tb = SummaryWriter(log_dir=dirname)

    # --- Initializations ---
    pretrained_epochs = agent.load_param(load_context=args.load_context)
    # model = CNNClassifier() #models.resnet18(pretrained=True)
    # model = load_model()

    '''
    # # --- Freeze layers and replace FC layer ---
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         if "fc" not in name:
    #             param.requires_grad = False
    # model.fc = torch.nn.Linear(512, 3)
    '''

    # Potential GPU optimization.
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, agent.net.parameters()), lr=args.lrate)
    criterion = torch.nn.MSELoss()
    train_loader = load_data("/home/kxa200005/data/train")
    validation_loader = load_data("/home/kxa200005/data/valid")

    train_iterations = 0
    val_iterations = 0
    # --- SGD Iterations ---
    for epoch in range(pretrained_epochs + 1, args.epochs):
        
        print("Starting Epoch: ", epoch)
        if (epoch % 10 == 0 and epoch != 0):
            agent.save_param(epoch=epoch)

        # Per epoch train loop.
        agent.net.train()
        for _, (rgb_input, sem_input, yhat, image_path) in enumerate(train_loader):
            yhat = yhat.view(-1, 3, 1)
            yhat = yhat.cuda()
            optimizer.zero_grad()
            # sem_input.to(device)
            # ypred = model(rgb_input.cuda())
            ypred = agent.select_action(rgb_input)
            ypred = ypred.view(-1, 3, 1)
            loss = criterion(ypred, yhat)
            loss.backward()
            optimizer.step()

            # Record training loss and accuracy
            tb.add_scalar("Train Loss", loss, train_iterations)
            steer, throttle, brake, average = accuracy(ypred, yhat)
            tb.add_scalar("Train Accuracy", average, train_iterations)

            tb.add_scalar("Steer Accuracy", steer, train_iterations)
            tb.add_scalar("Throttle Accuracy", throttle, train_iterations)
            tb.add_scalar("Brake Accuracy", brake, train_iterations)
            LOGGER.info('Epoch {}\t Train Accuracy: {:.2f}\t Steer Accuracy: {:.2f}\t Throttle Accuracy: {:.2f} \t Brake Accuracy: {}'.format(epoch, average, steer, throttle, brake))
            train_iterations += 1
        
        # After each train epoch, do validation before starting next train epoch.
        agent.net.eval()
        for _, (rgb_input, sem_input, yhat, image_path) in enumerate(validation_loader):
            yhat = yhat.view(-1, 3, 1)
            yhat = yhat.cuda()
            with torch.no_grad():
                # sem_input.to(device)
                # ypred = model(sem_input.cuda())
                ypred = agent.select_action(rgb_input)
                ypred = ypred.view(-1, 3, 1)
                loss = criterion(ypred, yhat)

            # Record validation loss and accuracy
            tb.add_scalar("Validation Loss", loss, val_iterations)
            steer, throttle, brake, average = accuracy(ypred, yhat)
            tb.add_scalar("Validation Accuracy", average, val_iterations)
            val_iterations += 1
            # tb.add_scalar("Steer Accuracy", steer, epoch)
            # tb.add_scalar("Throttle Accuracy", throttle, epoch)
            # tb.add_scalar("Brake Accuracy", brake, epoch)
    '''
    save_model(model)
    '''
    agent.save_param(epoch=epoch)

'''
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lrate', type=float, default=0.001)
    parser.add_argument('-e', '--epochs', type=int, default=450)
    parser.add_argument('-t', '--tbdir', type=str, default="")

    args = parser.parse_args()
    train(args)
'''
