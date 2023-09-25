from data.prediction.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_OPTIMIZER_CLASS,
    DEFAULT_SAVE_FREQUENCY,
    DEFAULT_TRUE_THRESHOLD,
)

CHECKPOINT_PATH = "clindev_model_checkpoints"


BATCH_SIZE = 1024  # DEFAULT_BATCH_SIZE
DEVICE = "mps"
EMBEDDING_DIM = 16
LR = 1e-4
MAX_ITEMS_PER_CAT = 5  # DEFAULT_MAX_ITEMS_PER_CAT
OPTIMIZER_CLASS = DEFAULT_OPTIMIZER_CLASS
SAVE_FREQUENCY = DEFAULT_SAVE_FREQUENCY
TRUE_THRESHOLD = DEFAULT_TRUE_THRESHOLD

SINGLE_SELECT_CATEGORICAL_FIELDS: list[str] = [
    "phase",
    "sponsor_type",
    # "comparator",
    # "facilities", ??
    # "countries" ??
]
Y1_CATEGORICAL_FIELDS: list[str] = [
    "design",
    "masking",
    "randomization",
]
MULTI_SELECT_CATEGORICAL_FIELDS: list[str] = [
    "conditions",
    "interventions",
    # "termination_reason",
]
TEXT_FIELDS: list[str] = []

# enrollment + duration have very low correlation, high covariance
# (low corr is perhaps why it doesn't offer much loss reduction)
QUANTITATIVE_FIELDS: list[str] = [
    "enrollment",
    "start_date",
]
Y2_FIELD = "duration"


# with start date
# INFO:root:Stage1 design Metrics: {'precision': 0.3601883086205383, 'recall': 0.28252967520760774, 'f1-score': 0.25608948897399253}
# INFO:root:Stage1 design Accuracy: 0.5561899038461539
# INFO:root:Stage1 masking Metrics: {'precision': 0.27554986048907154, 'recall': 0.26145247795739185, 'f1-score': 0.2324802204045405}
# INFO:root:Stage1 masking Accuracy: 0.5302483974358975
# INFO:root:Stage1 randomization Metrics: {'precision': 0.44463306265185776, 'recall': 0.28662344452298383, 'f1-score': 0.27285090366887976}
# INFO:root:Stage1 randomization Accuracy: 0.7072315705128205
# INFO:root:Stage2 MAE: 290.42245092147436

# (only 2000c)
# INFO:root:Stage1 design Metrics: {'precision': 0.41458479521617725, 'recall': 0.29075154497309463, 'f1-score': 0.27267735527793896}
# INFO:root:Stage1 design Accuracy: 0.5693359375
# INFO:root:Stage1 masking Metrics: {'precision': 0.3358596375427413, 'recall': 0.2867235204906746, 'f1-score': 0.2654897214192193}
# INFO:root:Stage1 masking Accuracy: 0.51416015625
# INFO:root:Stage1 randomization Metrics: {'precision': 0.54058229352347, 'recall': 0.3827675189722388, 'f1-score': 0.36977309664495445}
# INFO:root:Stage1 randomization Accuracy: 0.744140625
# INFO:root:Stage2 MAE: 162.5795135498047

# learnable embeddings, sized
# INFO:root:Stage1 design Metrics: {'precision': 0.2875904265960235, 'recall': 0.2652472435978736, 'f1-score': 0.22339165077887083}
# INFO:root:Stage1 design Accuracy: 0.310546875
# INFO:root:Stage1 masking Metrics: {'precision': 0.31644165358451076, 'recall': 0.2776932722945374, 'f1-score': 0.20352623877540613}
# INFO:root:Stage1 masking Accuracy: 0.31787109375
# INFO:root:Stage1 randomization Metrics: {'precision': 0.5785, 'recall': 0.37414965986394555, 'f1-score': 0.3552587237980494}
# INFO:root:Stage1 randomization Accuracy: 0.74169921875
# INFO:root:Stage2 MAE: 155.61978149414062

# with aux loss
# INFO:root:Stage1 design Metrics: {'0': {'precision': 0.19007751937984496, 'recall': 0.9781914893617021, 'f1-score': 0.3183037646040672}, '1': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}, '2': {'precision': 0.46112115732368897, 'recall': 0.04779756326148079, 'f1-score': 0.08661684782608679}, '3': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}, '4': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}, 'macro avg': {'precision': 0.1302397353407068, 'recall': 0.2051978105246366, 'f1-score': 0.08098412248603079}}
# INFO:root:Stage1 design Accuracy: 0.2044921875
# INFO:root:Stage1 masking Metrics: {'0': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}, '1': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}, '2': {'precision': 0.23510413186323645, 'recall': 0.881419624217119, 'f1-score': 0.3711974679092664}, '3': {'precision': 0.0782312925170068, 'recall': 0.06388888888888888, 'f1-score': 0.07033639143730837}, '4': {'precision': 0.12202688728024819, 'recall': 0.13111111111111112, 'f1-score': 0.12640599892876223}, 'macro avg': {'precision': 0.08707246233209828, 'recall': 0.2152839248434238, 'f1-score': 0.1135879716550674}}
# INFO:root:Stage1 masking Accuracy: 0.219921875
# INFO:root:Stage1 randomization Metrics: {'0': {'precision': 0.3584158415841584, 'recall': 0.0923469387755102, 'f1-score': 0.14685598377281914}, '1': {'precision': 0.20754716981132076, 'recall': 0.011827956989247311, 'f1-score': 0.022380467955238962}, '2': {'precision': 0.7270192109068374, 'recall': 0.957687074829932, 'f1-score': 0.8265617660873643}, 'macro avg': {'precision': 0.43099407410077223, 'recall': 0.3539539901982298, 'f1-score': 0.3319327392718075}}
# INFO:root:Stage1 randomization Accuracy: 0.70615234375
# INFO:root:Stage2 MAE: 304.4075927734375

# # with aux loss & smmoth
# INFO:root:Stage1 design Metrics: {'0': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}, '1': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}, '2': {'precision': 0.5218667449368947, 'recall': 0.9998125585754452, 'f1-score': 0.685780406274106}, '3': {'precision': 0.42105263157894735, 'recall': 0.005970149253731343, 'f1-score': 0.011773362766740222}, '4': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}, 'macro avg': {'precision': 0.1885838753031684, 'recall': 0.20115654156583532, 'f1-score': 0.13951075380816924}}
# INFO:root:Stage1 design Accuracy: 0.5216796875
# INFO:root:Stage1 masking Metrics: {'0': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}, '1': {'precision': 0.5755842864246643, 'recall': 0.5065645514223195, 'f1-score': 0.5388733705772806}, '2': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}, '3': {'precision': 0.034898681247989706, 'recall': 0.6027777777777777, 'f1-score': 0.06597750076010936}, '4': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}, 'macro avg': {'precision': 0.1220965935345308, 'recall': 0.22186846584001946, 'f1-score': 0.12097017426747798}}
# INFO:root:Stage1 masking Accuracy: 0.247265625
# INFO:root:Stage1 randomization Metrics: {'0': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}, '1': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}, '2': {'precision': 0.7181240840254031, 'recall': 0.9993201903467029, 'f1-score': 0.835702103467879}, 'macro avg': {'precision': 0.23937469467513436, 'recall': 0.33310673011556763, 'f1-score': 0.2785673678226263}}
# INFO:root:Stage1 randomization Accuracy: 0.7177734375
# INFO:root:Stage2 MAE: 301.7379638671875

# higher weight decay
# INFO:root:Stage1 design Metrics: {'precision': 0.3044067955148667, 'recall': 0.211263168118451, 'f1-score': 0.16326977421929}
# INFO:root:Stage1 design Accuracy: 0.52294921875
# INFO:root:Stage1 masking Metrics: {'precision': 0.2604082484174232, 'recall': 0.2567331770846363, 'f1-score': 0.19891699273139474}
# INFO:root:Stage1 masking Accuracy: 0.2978515625
# INFO:root:Stage1 randomization Metrics: {'precision': 0.4572053831715986, 'recall': 0.3592052870167855, 'f1-score': 0.3332139953930379}
# INFO:root:Stage1 randomization Accuracy: 0.7275390625
# INFO:root:Stage2 MAE: 156.21859741210938

# 10,000
# INFO:root:Stage1 design Metrics: {'precision': 0.18245140297881934, 'recall': 0.18110410312022598, 'f1-score': 0.157358222229351}
# INFO:root:Stage1 design Accuracy: 0.5765224358974359
# INFO:root:Stage1 masking Metrics: {'precision': 0.1314500884096779, 'recall': 0.17943782271280673, 'f1-score': 0.13533546050584264}
# INFO:root:Stage1 masking Accuracy: 0.4795673076923077
# INFO:root:Stage1 randomization Metrics: {'precision': 0.40118861024033436, 'recall': 0.27401801140094983, 'f1-score': 0.24998637689635714}
# INFO:root:Stage1 randomization Accuracy: 0.7006209935897436
# INFO:root:Stage2 MAE: 351.2637970753205
