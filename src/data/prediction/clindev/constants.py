from data.prediction.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_OPTIMIZER_CLASS,
    DEFAULT_SAVE_FREQUENCY,
    DEFAULT_TRUE_THRESHOLD,
)

CHECKPOINT_PATH = "clindev_model_checkpoints"


BATCH_SIZE = 32  # DEFAULT_BATCH_SIZE
DEVICE = "mps"
EMBEDDING_DIM = 16
LR = 1e-4
MAX_ITEMS_PER_CAT = 10  # DEFAULT_MAX_ITEMS_PER_CAT
OPTIMIZER_CLASS = DEFAULT_OPTIMIZER_CLASS
SAVE_FREQUENCY = DEFAULT_SAVE_FREQUENCY
TRUE_THRESHOLD = DEFAULT_TRUE_THRESHOLD

SINGLE_SELECT_CATEGORICAL_FIELDS: list[str] = [
    "phase",
    "sponsor_type",
    "enrollment",
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


QUANTITATIVE_FIELDS: list[str] = [
    # "enrollment",
    "start_date",
]

# enrollment + duration have very low correlation, high covariance
# (low corr is perhaps why it doesn't offer much loss reduction)
QUANTITATIVE_TO_CATEGORY_FIELDS: list[str] = [
    "enrollment",
    "duration",
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


# LR = 1e-4
# INFO:root:Stage1 design Metrics: {'0': {'precision': 0.9876543209876543, 'recall': 0.1702127659574468, 'f1-score': 0.29038112522685994}, '1': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}, '2': {'precision': 0.5503326426000654, 'recall': 0.9458294283036551, 'f1-score': 0.6958080529509094}, '3': {'precision': 0.3713163064833006, 'recall': 0.141044776119403, 'f1-score': 0.2044348296376416}, '4': {'precision': 0.7941176470588235, 'recall': 0.11454545454545455, 'f1-score': 0.20021186440677946}, 'macro avg': {'precision': 0.5406841834259688, 'recall': 0.27432648498519185, 'f1-score': 0.2781671744444381}}
# INFO:root:Stage1 design Accuracy: 0.5609375
# INFO:root:Stage1 masking Metrics: {'0': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}, '1': {'precision': 0.4736481861738535, 'recall': 0.9085339168490153, 'f1-score': 0.6226754649070181}, '2': {'precision': 0.24789915966386555, 'recall': 0.049269311064718165, 'f1-score': 0.08220132358063366}, '3': {'precision': 0.06781750924784218, 'recall': 0.1527777777777778, 'f1-score': 0.09393680614859053}, '4': {'precision': 0.37433155080213903, 'recall': 0.07734806629834254, 'f1-score': 0.12820512820512794}, 'macro avg': {'precision': 0.23273928117754003, 'recall': 0.23758581439797072, 'f1-score': 0.18540374456827408}}
# INFO:root:Stage1 masking Accuracy: 0.42919921875
# INFO:root:Stage1 randomization Metrics: {'0': {'precision': 0.8744113029827315, 'recall': 0.2841836734693878, 'f1-score': 0.4289564882556792}, '1': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}, '2': {'precision': 0.7611784371082323, 'recall': 0.9906186267845003, 'f1-score': 0.8608731612217161}, 'macro avg': {'precision': 0.5451965800303212, 'recall': 0.4249341000846294, 'f1-score': 0.4299432164924651}}
# INFO:root:Stage1 randomization Accuracy: 0.76591796875

# remove rare y1 outputs from training set
# INFO:root:Stage1 design Metrics: {'0': {'precision': 0.5068087625814092, 'recall': 0.40956937799043064, 'f1-score': 0.45302990209049965}, '1': {'precision': 0.6752781211372064, 'recall': 0.8678316123907863, 'f1-score': 0.7595411887382684}, '2': {'precision': 0.24295010845986983, 'recall': 0.06037735849056604, 'f1-score': 0.09671848013816893}, 'macro avg': {'precision': 0.4750123307261618, 'recall': 0.4459261162905943, 'f1-score': 0.436429856988979}}
# INFO:root:Stage1 design Accuracy: 0.62802734375
# INFO:root:Stage1 masking Metrics: {'0': {'precision': 0.8764845605700713, 'recall': 0.1868354430379747, 'f1-score': 0.30801335559265414}, '1': {'precision': 0.4965493834144134, 'recall': 0.95, 'f1-score': 0.6522029868489481}, '2': {'precision': 0.7748344370860927, 'recall': 0.1465553235908142, 'f1-score': 0.24648876404494355}, '3': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}, '4': {'precision': 0.22580645161290322, 'recall': 0.14166666666666666, 'f1-score': 0.17410387710314512}, 'macro avg': {'precision': 0.4747349665366961, 'recall': 0.2850114866590911, 'f1-score': 0.27616179671793817}}
# INFO:root:Stage1 masking Accuracy: 0.510546875
# INFO:root:Stage1 randomization Metrics: {'0': {'precision': 0.8791208791208791, 'recall': 0.12276214833759591, 'f1-score': 0.21543985637342888}, '1': {'precision': 0.5961538461538461, 'recall': 0.04078947368421053, 'f1-score': 0.07635467980295554}, '2': {'precision': 0.7535047907211296, 'recall': 0.9928239202657807, 'f1-score': 0.856766055045871}, 'macro avg': {'precision': 0.7429265053319517, 'recall': 0.3854585140958624, 'f1-score': 0.3828535304074185}}
# INFO:root:Stage1 randomization Accuracy: 0.7560546875
# INFO:root:Stage2 MAE: 146.845751953125


# 2000, batch size of 32 + batch norm + wider + higher weight decay
# INFO:root:Stage1 design Metrics: {'0': {'precision': 0.5039808917197452, 'recall': 0.34590163934426227, 'f1-score': 0.4102397926117947}, '1': {'precision': 0.7935799625737728, 'recall': 0.8842020850040097, 'f1-score': 0.8364436352602027}, '2': {'precision': 0.6260920209668026, 'recall': 0.5795148247978437, 'f1-score': 0.601903695408734}, 'macro avg': {'precision': 0.6412176250867735, 'recall': 0.6032061830487052, 'f1-score': 0.6161957077602438}}
# INFO:root:Stage1 design Accuracy: 0.7279233870967742
# INFO:root:Stage1 masking Metrics: {'0': {'precision': 0.33764553686934023, 'recall': 0.15263157894736842, 'f1-score': 0.21022956101490092}, '1': {'precision': 0.6708860759493671, 'recall': 0.8688524590163934, 'f1-score': 0.7571428571428567}, '2': {'precision': 0.5143658023826209, 'recall': 0.6142259414225941, 'f1-score': 0.5598779557589622}, '3': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}, '4': {'precision': 0.3551912568306011, 'recall': 0.15568862275449102, 'f1-score': 0.21648626144879224}, 'macro avg': {'precision': 0.37561773440638585, 'recall': 0.3582797204281694, 'f1-score': 0.3487473270731024}}
# INFO:root:Stage1 masking Accuracy: 0.5881048387096774
# INFO:root:Stage1 randomization Metrics: {'0': {'precision': 0.6313269493844049, 'recall': 0.5381924198250729, 'f1-score': 0.5810513062637703}, '1': {'precision': 0.4838709677419355, 'recall': 0.11920529801324503, 'f1-score': 0.19128586609989343}, '2': {'precision': 0.8441731141199227, 'recall': 0.9373154362416107, 'f1-score': 0.8883093753975315}, 'macro avg': {'precision': 0.6531236770820877, 'recall': 0.5315710513599762, 'f1-score': 0.5535488492537318}}
# INFO:root:Stage1 randomization Accuracy: 0.8060483870967742
# INFO:root:Stage2 MAE: 145.19381300403225

# lower LR (1e-3), no dropout
# INFO:root:Stage1 design Metrics: {'precision': 0.6007328782992817, 'recall': 0.5247468413229283, 'f1-score': 0.5450438363078526}
# INFO:root:Stage1 design Accuracy: 0.6814516129032258
# INFO:root:Stage1 masking Metrics: {'precision': 0.3934667220999648, 'recall': 0.3331720780161459, 'f1-score': 0.32092576577565735}
# INFO:root:Stage1 masking Accuracy: 0.5430443548387097
# INFO:root:Stage1 randomization Metrics: {'precision': 0.5752425923905847, 'recall': 0.43675346910274143, 'f1-score': 0.45396077515158284}
# INFO:root:Stage1 randomization Accuracy: 0.777016129032258
# INFO:root:Stage2 MAE: 155.0238029233871
# INFO:root:Saved checkpoint checkpoint_495.pt

# LR = 1e-4 no dropout early stopping
# INFO:root:Stage1 design Metrics: {'precision': 0.9534018783108008, 'recall': 0.9447741797276228, 'f1-score': 0.9490162443608342}
# INFO:root:Stage1 design Accuracy: 0.9602822580645162
# INFO:root:Stage1 masking Metrics: {'precision': 0.656033849922445, 'recall': 0.6207700903378501, 'f1-score': 0.6239451844747066}
# INFO:root:Stage1 masking Accuracy: 0.8418346774193548
# INFO:root:Stage1 randomization Metrics: {'precision': 0.8552756044490288, 'recall': 0.7361480323623094, 'f1-score': 0.7572132673289986}
# INFO:root:Stage1 randomization Accuracy: 0.9251008064516129
# INFO:root:Stage2 MAE: 62.788142641129035
