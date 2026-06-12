# hyperparams and vocab stuff

N_DIGITS = 3

VOCAB_SIZE = 14
CHAR_TO_ID = {str(i): i for i in range(10)}
CHAR_TO_ID['+'] = 10
CHAR_TO_ID['='] = 11
PAD_TOKEN = 12
EOS_TOKEN = 13
ID_TO_CHAR = {v: k for k, v in CHAR_TO_ID.items()}
ID_TO_CHAR[PAD_TOKEN] = '<PAD>'
ID_TO_CHAR[EOS_TOKEN] = '<EOS>'

# "012+034=0046<EOS>" -> 13 tokens
# input: 3 digits + '+' + 3 digits + '=' = 8
# answer: 4 digits + EOS = 5
SEQ_LEN = 13
ANSWER_START = 8

N_LAYERS = 2
N_HEADS = 3
D_MODEL = 128
D_HEAD = 42
D_MLP = 512
ACT_FN = "relu"
NORMALIZATION_TYPE = None

BATCH_SIZE = 64
LR = 1e-3
LR_MIN = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
TOTAL_STEPS = 12000
GRAD_CLIP_NORM = 1.0

EVAL_EVERY = 500
LOG_EVERY = 100

TRAIN_SIZE = 300_000
TEST_SIZE_PER_CATEGORY = 10_000

SEED = 42
DEVICE = "cpu"
