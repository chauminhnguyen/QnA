<<<<<<< HEAD
from model import QuestionAnswering, Model
=======
from model import QuestionAnswering
>>>>>>> e76bba43cbb7f68efd0ac987bdd2ec9f36771114
import argparse

question_path = "/home/lake/Project/Vi_ICE/testing_section/data/corpus/multiple_ps/2021/preprocessed_Milestone_1&2_New_York_&_Bitcoin.xlsx"
doc_dir = '/home/lake/Project/Vi_ICE/testing_section/data/corpus/multiple_ps/2021/test'

parser = argparse.ArgumentParser(description='Question Answering')
parser.add_argument("command", metavar="<command>", help="'test' or 'evaluate'")
parser.add_argument('-q', '--question_path', type=str, default=question_path, help='Path to the .xlsx question file for evaluation.')
parser.add_argument('-qm', '--question_name', type=str, default='NewYork', help='Name of the question file for evaluation.')
parser.add_argument('-dd', '--doc_dir', type=str, default=doc_dir, help='Path to the directory of the documents.')
<<<<<<< HEAD
parser.add_argument('-m', '--model_name', type=str, default="./models/roberta-large-squad2", help='Name of the model.')
parser.add_argument('-d', '--device', type=str, default=2, help='Name of the device.')
args = parser.parse_args()

# qa = QuestionAnswering(args.doc_dir, args.model_name, args.device)
# qa.evaluate(args.question_path, args.question_name)

qa = Model(args.command, args.doc_dir, args.model_name, args.device, question_path=args.question_path, data_name=args.question_name)
qa.run()
# qa = Model(args.command, args.doc_dir, args.model_name, args.device)
# qa.run()

# 60
# 100
# Accuracy: 0.7606646845200541
# Counter({3: 97, 2: 3})

# 63
# 100
# Accuracy: 0.8099413663878867
# Counter({3: 100})

# 62
# 100
# Accuracy: 0.804665222341756
# Counter({3: 100})

# 55
# 100
# Accuracy: 0.7173269107392741
# Counter({3: 98, 2: 2})
=======
parser.add_argument('-m', '--model_name', type=str, default="deepset/roberta-large-squad2", help='Name of the model.')
parser.add_argument('-d', '--device', type=int, default=2, help='Name of the device.')
args = parser.parse_args()

qa = QuestionAnswering(args.doc_dir, args.model_name, args.device)
qa.evaluate(args.question_path, args.question_name)
>>>>>>> e76bba43cbb7f68efd0ac987bdd2ec9f36771114
