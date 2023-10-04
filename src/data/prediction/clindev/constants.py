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
    # "facilities", ??
    # "countries" ??
]
Y1_CATEGORICAL_FIELDS: list[str] = [
    "design",
    "masking",
    "randomization",
    "comparison_type",
    # "hypothesis_type"
    # "termination_reason",
    # dropout_count
]
MULTI_SELECT_CATEGORICAL_FIELDS: list[str] = [
    "conditions",  # typically only the specific condition
    "mesh_conditions",  # normalized; includes ancestors
    "interventions",
]
TEXT_FIELDS: list[str] = []


QUANTITATIVE_FIELDS: list[str] = [
    "start_date",
]

# Side note: enrollment + duration have very low correlation, high covariance
# (low corr is perhaps why it doesn't offer much loss reduction)
QUANTITATIVE_TO_CATEGORY_FIELDS: list[str] = [
    "enrollment",
    "duration",
    # dropout_count
]
Y2_FIELD = "duration"  # TODO: include distance measure


# new data (mesh conditions, comparator type) - BIG improvement (stage2 CEL was 0.32 before)
# INFO:__main__:Stage1 design: {"0": {"precision": 0.9843843843843844, "recall": 0.9669616519174041, "f1-score": 0.9755952380952375}, "1": {"precision": 0.9891186071817193, "recall": 0.9926677067082683, "f1-score": 0.9908899789768739}, "2": {"precision": 0.9829857299670691, "recall": 0.9867768595041322, "f1-score": 0.9848776464118773}, "macro avg": {"precision": 0.9854962405110577, "recall": 0.9821354060432682, "f1-score": 0.9837876211613296}}
# INFO:__main__:Stage1 design: 0.9871975806451613
# INFO:__main__:Stage1 masking: {"0": {"precision": 0.9020141535111595, "recall": 0.9495702005730658, "f1-score": 0.9251814628699044}, "1": {"precision": 0.985723698660224, "recall": 0.9885462555066079, "f1-score": 0.987132959419333}, "2": {"precision": 0.9076595744680851, "recall": 0.9543624161073826, "f1-score": 0.9304252998909481}, "3": {"precision": 0.888268156424581, "recall": 0.6694736842105263, "f1-score": 0.7635054021608638}, "4": {"precision": 0.8600973236009732, "recall": 0.7643243243243243, "f1-score": 0.8093875214653685}, "macro avg": {"precision": 0.9087525813330046, "recall": 0.8652553761443814, "f1-score": 0.8831265291612835}}
# INFO:__main__:Stage1 masking: 0.9378024193548387
# INFO:__main__:Stage1 randomization: {"0": {"precision": 0.950228832951945, "recall": 0.9916417910447761, "f1-score": 0.970493718959976}, "1": {"precision": 0.9488721804511279, "recall": 0.8887323943661972, "f1-score": 0.9178181818181812}, "2": {"precision": 0.9930731317437059, "recall": 0.9893828798938288, "f1-score": 0.9912245712006378}, "macro avg": {"precision": 0.9640580483822596, "recall": 0.9565856884349341, "f1-score": 0.9598454906595983}}
# INFO:__main__:Stage1 randomization: 0.9825604838709677
# INFO:__main__:Stage1 comparison_type: {"0": {"precision": 0.9757113115891742, "recall": 0.9959858323494687, "f1-score": 0.985744332788034}, "1": {"precision": 0.9722991689750693, "recall": 0.9915254237288136, "f1-score": 0.9818181818181814}, "2": {"precision": 0.7435897435897436, "recall": 0.15675675675675677, "f1-score": 0.2589285714285712}, "3": {"precision": 0.9090909090909091, "recall": 0.16666666666666666, "f1-score": 0.2816901408450701}, "4": {"precision": 0.9743452699091395, "recall": 0.9961748633879781, "f1-score": 0.9851391515806532}, "5": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "macro avg": {"precision": 0.7625060671923393, "recall": 0.5511849238149473, "f1-score": 0.5822200630767517}}
# INFO:__main__:Stage1 comparison_type: 0.9735887096774194
# INFO:__main__:Stage2: {"0": {"precision": 0.6451388888888889, "recall": 0.78895966029724, "f1-score": 0.7098376313276021}, "1": {"precision": 0.486331569664903, "recall": 0.6027322404371585, "f1-score": 0.5383113714006827}, "2": {"precision": 0.5856443719412724, "recall": 0.4835016835016835, "f1-score": 0.5296938399114712}, "3": {"precision": 0.4344073647871116, "recall": 0.6371308016877637, "f1-score": 0.5165925419089971}, "4": {"precision": 0.5444664031620553, "recall": 0.58, "f1-score": 0.5616717635066253}, "5": {"precision": 0.5540201005025126, "recall": 0.5547169811320755, "f1-score": 0.5543683218101817}, "6": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "7": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "8": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "9": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "macro avg": {"precision": 0.32500086989467436, "recall": 0.3647041367055921, "f1-score": 0.341047546986556}}
# INFO:__main__:Stage2: 0.5469758064516129

# 20,000 at completion (given eval, would probably continue to learn)
# INFO:__main__:Stage1 design: {"0": {"precision": 0.7569049343126099, "recall": 0.6778138026864289, "f1-score": 0.7151793568566116}, "1": {"precision": 0.9181534848005781, "recall": 0.9476456876456877, "f1-score": 0.9326664984285021}, "2": {"precision": 0.9083458772369961, "recall": 0.8740293703480185, "f1-score": 0.8908572717393527}, "macro avg": {"precision": 0.8611347654500614, "recall": 0.833162953560045, "f1-score": 0.8462343756748222}}
# INFO:__main__:Stage1 design: 0.90022
# INFO:__main__:Stage1 masking: {"0": {"precision": 0.4917950266787476, "recall": 0.5946439440048692, "f1-score": 0.5383513334802726}, "1": {"precision": 0.8593845332519829, "recall": 0.9401960784313725, "f1-score": 0.8979758536876913}, "2": {"precision": 0.6524042706911777, "recall": 0.7728958630527818, "f1-score": 0.7075570259446277}, "3": {"precision": 0.6119016817593791, "recall": 0.26875, "f1-score": 0.3734701934465057}, "4": {"precision": 0.5011037527593819, "recall": 0.024356223175965665, "f1-score": 0.0464545175483474}, "macro avg": {"precision": 0.6233178530281338, "recall": 0.5201684217329978, "f1-score": 0.5127617848214889}}
# INFO:__main__:Stage1 masking: 0.72743
# INFO:__main__:Stage1 randomization: {"0": {"precision": 0.8832086061739944, "recall": 0.8842425661437603, "f1-score": 0.8837252837252833}, "1": {"precision": 0.7806619078520441, "recall": 0.5966522008679479, "f1-score": 0.6763651697238029}, "2": {"precision": 0.9413851164844871, "recall": 0.966406914139983, "f1-score": 0.9537319276266109}, "macro avg": {"precision": 0.8684185435035086, "recall": 0.8157672270505637, "f1-score": 0.837940793691899}}
# INFO:__main__:Stage1 randomization: 0.91904
# INFO:__main__:Stage1 comparison_type: {"0": {"precision": 0.8391112626988574, "recall": 0.8859935691318328, "f1-score": 0.8619153674832957}, "1": {"precision": 0.8899653401261118, "recall": 0.9034336583298007, "f1-score": 0.8966489260996692}, "2": {"precision": 0.6891891891891891, "recall": 0.05454545454545454, "f1-score": 0.10109018830525258}, "3": {"precision": 1.0, "recall": 0.004838709677419355, "f1-score": 0.009630818619582654}, "4": {"precision": 0.8634916080906613, "recall": 0.865727024306055, "f1-score": 0.8646078713013496}, "5": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "macro avg": {"precision": 0.7136262333508032, "recall": 0.4524230693317604, "f1-score": 0.4556488619681916}}
# INFO:__main__:Stage1 comparison_type: 0.85957
# INFO:__main__:Stage2: {"0": {"precision": 0.5632741563897548, "recall": 0.7742921013412817, "f1-score": 0.6521377579038201}, "1": {"precision": 0.4262310941984585, "recall": 0.5793118562349332, "f1-score": 0.49111920332936926}, "2": {"precision": 0.40541425098185896, "recall": 0.5136848341232227, "f1-score": 0.45317236333228755}, "3": {"precision": 0.39584703561026024, "recall": 0.3558277027027027, "f1-score": 0.3747720499933278}, "4": {"precision": 0.2222222222222222, "recall": 0.00023515579071134627, "f1-score": 0.0004698144233027933}, "5": {"precision": 0.10714285714285714, "recall": 0.001000834028356964, "f1-score": 0.0019831432821021135}, "6": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "7": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "8": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "9": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "macro avg": {"precision": 0.21201316165454118, "recall": 0.22243524842212087, "f1-score": 0.19736543322642094}}
# INFO:__main__:Stage2: 0.46891

# 40,000 records, 100 epochs
# INFO:__main__:Stage1 design: {"0": {"precision": 0.6038461538461538, "recall": 0.4350692785475394, "f1-score": 0.505748403221327}, "1": {"precision": 0.8035090499204126, "recall": 0.8970008844576667, "f1-score": 0.8476849371796767}, "2": {"precision": 0.7613229453764188, "recall": 0.6413345521023766, "f1-score": 0.6961966282658087}, "macro avg": {"precision": 0.722892716380995, "recall": 0.6578015717025275, "f1-score": 0.6832099895556042}}
# INFO:__main__:Stage1 design: 0.778735
# INFO:__main__:Stage1 masking: {"0": {"precision": 0.5172424035405591, "recall": 0.4528998699609883, "f1-score": 0.4829374488678116}, "1": {"precision": 0.7200941082908079, "recall": 0.8920314023295192, "f1-score": 0.7968939800235605}, "2": {"precision": 0.4420616658996779, "recall": 0.5091969255234562, "f1-score": 0.47326025372582786}, "3": {"precision": 0.4603330068560235, "recall": 0.1010752688172043, "f1-score": 0.16575559865984807}, "4": {"precision": 0.29411764705882354, "recall": 0.0020497803806734994, "f1-score": 0.004071187623589611}, "macro avg": {"precision": 0.4867697663291784, "recall": 0.3914506494023683, "f1-score": 0.3845836937801275}}
# INFO:__main__:Stage1 masking: 0.622625
# INFO:__main__:Stage1 randomization: {"0": {"precision": 0.6515359926639156, "recall": 0.5804738562091504, "f1-score": 0.6139554979477204}, "1": {"precision": 0.5753424657534246, "recall": 0.2008888888888889, "f1-score": 0.2977975779632298}, "2": {"precision": 0.8386717472773461, "recall": 0.9550606852601052, "f1-score": 0.8930901835006742}, "macro avg": {"precision": 0.6885167352315621, "recall": 0.5788078101193815, "f1-score": 0.6016144198038748}}
# INFO:__main__:Stage1 randomization: 0.7972
# INFO:__main__:Stage1 comparison_type: {"0": {"precision": 0.6229092550310625, "recall": 0.6722297829668362, "f1-score": 0.6466304250466794}, "1": {"precision": 0.6756529428257165, "recall": 0.6830480758748501, "f1-score": 0.6793303842484164}, "2": {"precision": 0.35294117647058826, "recall": 0.00384, "f1-score": 0.007597340930674243}, "3": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "4": {"precision": 0.6195614755183645, "recall": 0.7212237977805179, "f1-score": 0.6665384670170784}, "5": {"precision": 0.48850784385260854, "recall": 0.08920719520319786, "f1-score": 0.1508647400146468}, "macro avg": {"precision": 0.4599287822830567, "recall": 0.3615914753042336, "f1-score": 0.3584935595429159}}
# INFO:__main__:Stage1 comparison_type: 0.631985
# INFO:__main__:Stage2: {"0": {"precision": 0.5596038786878481, "recall": 0.8060025258153183, "f1-score": 0.6605741422874358}, "1": {"precision": 0.42564233997565193, "recall": 0.5317058529264632, "f1-score": 0.47279876871613696}, "2": {"precision": 0.3884996734548477, "recall": 0.3703693222017829, "f1-score": 0.379217919184016}, "3": {"precision": 0.3809386697379787, "recall": 0.12642140468227425, "f1-score": 0.18984072320275472}, "4": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "5": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "6": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "7": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "8": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "9": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "macro avg": {"precision": 0.17546845618563262, "recall": 0.1834499105625839, "f1-score": 0.17024315533903436}}
# INFO:__main__:Stage2: 0.482765
# INFO:__main__:Stage2: 1.175455

# 40,000 records, 150 epochs
# INFO:__main__:Stage1 design: {"0": {"precision": 0.6056683214981462, "recall": 0.4605973715651135, "f1-score": 0.5232640208480368}, "1": {"precision": 0.8463173070425073, "recall": 0.9061381999437231, "f1-score": 0.8752067463368037}, "2": {"precision": 0.8021017481830681, "recall": 0.7466630096909855, "f1-score": 0.773390151515151}, "macro avg": {"precision": 0.7513624589079072, "recall": 0.704466193733274, "f1-score": 0.7239536395666639}}
# INFO:__main__:Stage1 design: 0.815915
# INFO:__main__:Stage1 masking: {"0": {"precision": 0.5377120355411955, "recall": 0.5540902588112888, "f1-score": 0.5457783015846577}, "1": {"precision": 0.7573859862502048, "recall": 0.9010981116584564, "f1-score": 0.8230155270915638}, "2": {"precision": 0.5222584636148915, "recall": 0.6153458786111847, "f1-score": 0.5649936119729873}, "3": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "4": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "macro avg": {"precision": 0.36347129708125836, "recall": 0.414106849816186, "f1-score": 0.3867574881298418}}
# INFO:__main__:Stage1 masking: 0.66161
# INFO:__main__:Stage1 randomization: {"0": {"precision": 0.7534103442270854, "recall": 0.7070134822279722, "f1-score": 0.7294749118295882}, "1": {"precision": 0.6546516597805955, "recall": 0.3890770533446232, "f1-score": 0.48807690265016684}, "2": {"precision": 0.8789502864535206, "recall": 0.9534815451258368, "f1-score": 0.91470019541647}, "macro avg": {"precision": 0.7623374301537339, "recall": 0.6831906935661441, "f1-score": 0.7107506699654084}}
# INFO:__main__:Stage1 randomization: 0.84158
# INFO:__main__:Stage1 comparison_type: {"0": {"precision": 0.6605085554570864, "recall": 0.6940047274550534, "f1-score": 0.6768424729304919}, "1": {"precision": 0.7577130757014293, "recall": 0.7802660270388138, "f1-score": 0.7688241929419343}, "2": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "3": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "4": {"precision": 0.6810066772295156, "recall": 0.7420841683366733, "f1-score": 0.7102347334720188}, "5": {"precision": 0.6010066346373828, "recall": 0.34968386023294507, "f1-score": 0.442125636386586}, "macro avg": {"precision": 0.45003915717090237, "recall": 0.4276731305105809, "f1-score": 0.43300450595517187}}
# INFO:__main__:Stage1 comparison_type: 0.688105
# INFO:__main__:Stage2: {"0": {"precision": 0.5593111088729983, "recall": 0.8251708766716196, "f1-score": 0.6667146887568276}, "1": {"precision": 0.42689842777444786, "recall": 0.5558735241144687, "f1-score": 0.4829228856802589}, "2": {"precision": 0.3899398999319063, "recall": 0.37348645966255495, "f1-score": 0.3815358767127251}, "3": {"precision": 0.38853161843515543, "recall": 0.034523809523809526, "f1-score": 0.06341292749059725}, "4": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "5": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "6": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "7": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "8": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "9": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "macro avg": {"precision": 0.1764681055014508, "recall": 0.17890546699724527, "f1-score": 0.15945863786404088}}
# INFO:__main__:Stage2: 0.486035
# INFO:__main__:Stage2: 1.191

# 40,000 records, 200 epochs
# INFO:__main__:Stage1 design: {"0": {"precision": 0.6123668165301472, "recall": 0.47505972288580983, "f1-score": 0.5350445287486206}, "1": {"precision": 0.8494470156879245, "recall": 0.908972503617945, "f1-score": 0.8782022402087957}, "2": {"precision": 0.8233761619989343, "recall": 0.7628085573230938, "f1-score": 0.7919359890657474}, "macro avg": {"precision": 0.7617299980723353, "recall": 0.7156135946089496, "f1-score": 0.7350609193410547}}
# INFO:__main__:Stage1 design: 0.823595
# INFO:__main__:Stage1 masking: {"0": {"precision": 0.5453556263269639, "recall": 0.5345038366497594, "f1-score": 0.539875205254515}, "1": {"precision": 0.7756167852932186, "recall": 0.9114532019704433, "f1-score": 0.8380664795111934}, "2": {"precision": 0.4945866910730629, "recall": 0.5872250198780811, "f1-score": 0.5369394258848613}, "3": {"precision": 0.5672630881378397, "recall": 0.18398710370768404, "f1-score": 0.277854418566907}, "4": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "macro avg": {"precision": 0.476564438166217, "recall": 0.4434338324411935, "f1-score": 0.4385471058434954}}
# INFO:__main__:Stage1 masking: 0.666145
# INFO:__main__:Stage1 randomization: {"0": {"precision": 0.7484602463605823, "recall": 0.7281764946207272, "f1-score": 0.7381790570856624}, "1": {"precision": 0.6387888079724032, "recall": 0.35279424216765454, "f1-score": 0.45454793399699944}, "2": {"precision": 0.8833735577082025, "recall": 0.9565531808255469, "f1-score": 0.9185080783774487}, "macro avg": {"precision": 0.7568742040137293, "recall": 0.6791746392046428, "f1-score": 0.7037450231533703}}
# INFO:__main__:Stage1 randomization: 0.843325
# INFO:__main__:Stage1 comparison_type: {"0": {"precision": 0.6832126625092485, "recall": 0.7407391491190374, "f1-score": 0.7108138943490627}, "1": {"precision": 0.7581360640106132, "recall": 0.7974272320941894, "f1-score": 0.7772854304143151}, "2": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "3": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "4": {"precision": 0.7021307021307022, "recall": 0.7568984122090334, "f1-score": 0.7284866468842723}, "5": {"precision": 0.5728013029315961, "recall": 0.23423243423243423, "f1-score": 0.3324982273694158}, "macro avg": {"precision": 0.45271345526369333, "recall": 0.42154953794244904, "f1-score": 0.42484736650284427}}
# INFO:__main__:Stage1 comparison_type: 0.70451
# INFO:__main__:Stage2: {"0": {"precision": 0.5624618945758544, "recall": 0.8088268073408128, "f1-score": 0.6635134640937178}, "1": {"precision": 0.42780407720875374, "recall": 0.5370685342671335, "f1-score": 0.47624962293947465}, "2": {"precision": 0.3942672855506661, "recall": 0.37019951889061836, "f1-score": 0.38185453855472634}, "3": {"precision": 0.4057713347921225, "recall": 0.1417921146953405, "f1-score": 0.21014980344937453}, "4": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "5": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "6": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "7": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "8": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "9": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "macro avg": {"precision": 0.1790304592127397, "recall": 0.18578869751939053, "f1-score": 0.17317674290372934}}
# INFO:__main__:Stage2: 0.48659
# INFO:__main__:Stage2: 1.174635


# INFO:__main__:Starting batch 1096 out of 1097
# INFO:__main__:Training Stage1 design cp: {"0": {"precision": 0.5505202080832333, "recall": 0.30708705357142857, "f1-score": 0.39425419114486265}, "1": {"precision": 0.7402836945113292, "recall": 0.8809569729531157, "f1-score": 0.8045173005229362}, "2": {"precision": 0.6755313230221431, "recall": 0.4984912244688494, "f1-score": 0.5736626389331817}, "macro avg": {"precision": 0.6554450752055686, "recall": 0.5621784169977979, "f1-score": 0.5908113768669936}}
# INFO:__main__:Training Stage1 design accuracy: 0.716214676390155
# INFO:__main__:Training Stage1 masking cp: {"0": {"precision": 0.36059970480851394, "recall": 0.38872295882763436, "f1-score": 0.37413357691687665}, "1": {"precision": 0.6536870180717742, "recall": 0.8830417460592414, "f1-score": 0.7512489377274978}, "2": {"precision": 0.29212468001806957, "recall": 0.18789346246973365, "f1-score": 0.22869267947659977}, "3": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "4": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "macro avg": {"precision": 0.2612822805796715, "recall": 0.2919316334713219, "f1-score": 0.27081503882419483}}
# INFO:__main__:Training Stage1 masking accuracy: 0.5481597538742023
# INFO:__main__:Training Stage1 randomization cp: {"0": {"precision": 0.5359913620731025, "recall": 0.43179768041237115, "f1-score": 0.4782856938943007}, "1": {"precision": 0.4923936067783555, "recall": 0.11512832057631697, "f1-score": 0.18662190271138165}, "2": {"precision": 0.8017217057410835, "recall": 0.952866606690112, "f1-score": 0.8707841444592674}, "macro avg": {"precision": 0.6100355581975138, "recall": 0.4999308692262667, "f1-score": 0.51189724702165}}
# INFO:__main__:Training Stage1 randomization accuracy: 0.7547117137648132
# INFO:__main__:Training Stage1 comparison_type cp: {"0": {"precision": 0.49049276914836637, "recall": 0.5535555368986651, "f1-score": 0.5201195894830669}, "1": {"precision": 0.5625815987190541, "recall": 0.5839427256456149, "f1-score": 0.5730631704410006}, "2": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "3": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "4": {"precision": 0.5085934603692445, "recall": 0.6351884433743865, "f1-score": 0.5648851189985995}, "5": {"precision": 0.381294964028777, "recall": 0.005536693653695482, "f1-score": 0.01091489471245428}, "macro avg": {"precision": 0.3238271320442403, "recall": 0.2963705665953937, "f1-score": 0.2781637956058536}}
# INFO:__main__:Training Stage1 comparison_type accuracy: 0.513947128532361
# INFO:__main__:Training Stage2 cp: {"0": {"precision": 0.5811565359206126, "recall": 0.7276801938219261, "f1-score": 0.6462167374883773}, "1": {"precision": 0.4059834326008078, "recall": 0.5766433294437961, "f1-score": 0.47649351171105975}, "2": {"precision": 0.36264976270286414, "recall": 0.32554744525547447, "f1-score": 0.34309846772167746}, "3": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "4": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "5": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "6": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "7": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "8": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "9": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "macro avg": {"precision": 0.13497897312242846, "recall": 0.16298709685211968, "f1-score": 0.14658087169211145}}
# INFO:__main__:Training Stage2 accuracy: 0.47075546946216956
# INFO:__main__:Training Stage2 mae: 0.9837055606198724
# INFO:__main__:Evaluation Stage1 design cp: {"0": {"precision": 0.482367758186398, "recall": 0.39081632653061227, "f1-score": 0.43179255918827464}, "1": {"precision": 0.7157296272899558, "recall": 0.8356997971602435, "f1-score": 0.7710761378136959}, "2": {"precision": 0.5895249695493301, "recall": 0.40930232558139534, "f1-score": 0.4831544796605935}, "macro avg": {"precision": 0.5958741183418946, "recall": 0.5452728164240837, "f1-score": 0.562007725554188}}
# INFO:__main__:Evaluation Stage1 design accuracy: 0.6709625912408759
# INFO:__main__:Evaluation Stage1 masking cp: {"0": {"precision": 0.33458177278401996, "recall": 0.28480340063761955, "f1-score": 0.30769230769230715}, "1": {"precision": 0.6267744833782569, "recall": 0.8279136007595538, "f1-score": 0.7134383309470234}, "2": {"precision": 0.29918800749531543, "recall": 0.3031645569620253, "f1-score": 0.30116315624017553}, "3": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "4": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "macro avg": {"precision": 0.25210885273151845, "recall": 0.28317631167183976, "f1-score": 0.2644587589759012}}
# INFO:__main__:Evaluation Stage1 masking accuracy: 0.5135720802919708
# INFO:__main__:Evaluation Stage1 randomization cp: {"0": {"precision": 0.4796747967479675, "recall": 0.3439119170984456, "f1-score": 0.4006035458317611}, "1": {"precision": 0.42424242424242425, "recall": 0.07894736842105263, "f1-score": 0.1331220285261487}, "2": {"precision": 0.7758274152485596, "recall": 0.939935064935065, "f1-score": 0.8500330323717238}, "macro avg": {"precision": 0.5599148787463171, "recall": 0.4542647834848544, "f1-score": 0.4612528689098778}}
# INFO:__main__:Evaluation Stage1 randomization accuracy: 0.7304972627737226
# INFO:__main__:Evaluation Stage1 comparison_type cp: {"0": {"precision": 0.4464775846294602, "recall": 0.49426063470627957, "f1-score": 0.46915558404101854}, "1": {"precision": 0.473432518597237, "recall": 0.46919431279620855, "f1-score": 0.47130388786035393}, "2": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "3": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "4": {"precision": 0.4781734635267088, "recall": 0.5950679056468906, "f1-score": 0.5302547770700632}, "5": {"precision": 0.312, "recall": 0.04262295081967213, "f1-score": 0.07499999999999979}, "macro avg": {"precision": 0.28501392779223433, "recall": 0.2668576339948418, "f1-score": 0.25761904149523923}}
# INFO:__main__:Evaluation Stage1 comparison_type accuracy: 0.46293339416058393
# INFO:__main__:Evaluation Stage2 cp: {"0": {"precision": 0.5779896013864818, "recall": 0.6794567062818336, "f1-score": 0.6246293116903381}, "1": {"precision": 0.36363636363636365, "recall": 0.5658914728682171, "f1-score": 0.44275966641394954}, "2": {"precision": 0.31216111541440744, "recall": 0.2507778469197262, "f1-score": 0.2781228433402341}, "3": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "4": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "5": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "6": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "7": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "8": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "9": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "macro avg": {"precision": 0.12537870804372528, "recall": 0.14961260260697767, "f1-score": 0.13455118214445216}}
# INFO:__main__:Evaluation Stage2 accuracy: 0.4406934306569343
# INFO:__main__:Evaluation Stage2 mae: 1.0224680656934306


# INFO:__main__:Training Stage2 accuracy: 0.479369872379216
# INFO:__main__:Training Stage2 mae: 0.9765439835916135
# INFO:__main__:Evaluation Stage2 accuracy: 0.4336222627737226
# INFO:__main__:Evaluation Stage2 mae: 1.0238366788321167
# INFO:__main__:Training Stage2 accuracy: 0.48094804010938924
# INFO:__main__:Training Stage2 mae: 0.9766579307201458
# INFO:__main__:Evaluation Stage2 accuracy: 0.43396441605839414
# INFO:__main__:Evaluation Stage2 mae: 1.026117700729927

# INFO:__main__:Training Stage2 accuracy: 0.48357452142206014
# INFO:__main__:Training Stage2 mae: 0.9757748404740201

# INFO:__main__:Evaluation Stage2 accuracy: 0.4252965328467153
# INFO:__main__:Evaluation Stage2 mae: 1.041742700729927
