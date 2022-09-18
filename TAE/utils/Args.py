import  argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', type=str, default='YAGO')
args.add_argument('--time-interval', type=int, default=1)
args.add_argument('--lr-conv', type=float, default=0.001)
args.add_argument('--n-epochs-conv', type=int, default=50)
args.add_argument('--embedding-dim', type=int, default=200)
args.add_argument('--embedding-dim1', type=int, default=20)
args.add_argument('--hidden-dim', type=int, default=12800)
args.add_argument('--dropout-ta', type=float, default=0.3)#0.3
args.add_argument('--input-drop', type=float, default=0.2)#0.2
args.add_argument('--hidden-drop', type=float, default=0.4)#0.4
args.add_argument('--feat-drop', type=float, default=0.3)#0.3
args.add_argument('--batch-size-conv', type=int, default=2048)#2048 for YAGO, 2560 for WIKI, ICEWS05-15, 1280 for ICEWS14
args.add_argument('--pred', type=str, default='sub')
args.add_argument("--reg-para", type=float, default=0.01)
args.add_argument('--valid-epoch', type=int, default= 5)
args.add_argument('--count', type=int, default=8)#3
args.add_argument("--multi-step", action='store_true', default=True)
args.add_argument('--seed', type=int, default=42)#42

args, unknown = args.parse_known_args()
print(args)
