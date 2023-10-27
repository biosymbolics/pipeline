TEXT_FIELDS = frozenset(["title", "abstract"])
APPLICATIONS_TABLE = "applications"
ANNOTATIONS_TABLE = "annotations"
GPR_ANNOTATIONS_TABLE = "gpr_annotations"
GPR_PUBLICATIONS_TABLE = "gpr_publications"


ADHD = "Attention Deficit Hyperactivity Disorder"
ARTICULATION_DISORDER = "Articulation Disorder"
AMNESTIC_DISORDER = "Amnestic Disorder"
ANAESTHESIA = "Anaesthesia"
ANEMIA = "Anemia"
ANXIETY = "Anxiety"
ASPERGERS = "Aspergers"
AUTISM = "Autism"
BDI = "Bipolar I Disorder"
BDII = "Bipolar II Disorder"
BIPOLAR = "Bipolar Disorder"
BODY_DYSMORPHIA = "Body Dysmorphic Disorder"
BPD = "Borderline Personality Disorder"
BRIEF_PSYCHOTIC_DISORDER = "Brief Psychotic Disorder"
CELIAC = "Celiac Disease"
CYCLOTHYMIA = "Cyclothymia"
DEPRESSION = "Mental Depression"  # mental depression due to UMLS
DEVELOPMENTAL_DISORDER = "Developmental Disorder"
DELUSION = "Delusion"
DIARRHEA = "Diarrhea"
DYSTHYMIA = "Dysthymia"
EATING_DISORDER = "Eating Disorder"
GERD = "Gastroesophageal Reflux Disease"
GENERALIZED_ANXIETY_DISORDER = "Generalized Anxiety Disorder"
HEMATOLOGICAL_DISORDER = "Hematological Disorder"
HYPERSENSITIVITY = "Hypersensitivity"
INTERMITTENT_EXPLOSIVE_DISORDER = "Intermittent Explosive Disorder"
ISCHEMIA = "Ischemia"
MENTAL_DISORDER = "Mental Disorder"
MIGRAINE = "Migraine"
MOTOR_NEURON_DISEASE = "Motor Neuron Disease"
MOOD_DISRDER = "Mood Disorder"
NEURODEVELOPMENTAL_DISORDER = "Neurodevelopmental Disorder"
NPD = "Narcissistic Personality Disorder"
DISSASOCIATIVE_IDENTITY_DISORDER = "Dissociative identity disorder"
NEUROSIS = "Neurosis"
OCD = "Obsessive-Compulsive Disorder"
OPPOSITIONAL_DEFIANT_DISORDER = "Oppositional Defiant Disorder"
PAIN_DISORDER = "Pain Disorder"
PANIC = "Panic Disorder"
PATHOGENIC_PAIN = "Pathogenic Pain"
PDD = "Premenstrual Dysphoric Disorder"
PERSONALITY_DISORDER = "Personality Disorder"
PHOBIA = "Phobia"
PHOTOSENSITIVITY = "Photosensitivity"
PSYCHIATRIC_DISORDER = "Psychiatric Disorder"
PSYCHOTIC_DISORDER = "Psychotic Disorder"
PTSD = "Post-Traumatic Stress Disorder"
REPRODUCTIVE_DISORDER = "Reproductive Disorder"
SAD = "Seasonal Affective Disorder"
SCHIZOPHRENIA = "Schizophrenia"
SCHIZOPHRENIFORM_DISORDER = "Schizophreniform Disorder"
SCHIZOAFFECTIVE_DISORDER = "Schizoaffective Disorder"
SEXUAL_DISORDER = "Sexual Disorder"
SLEEP_APNEA = "Sleep Apnea"
SLEEP_DISORDER = "Sleep Disorder"
SOCIOPATHY = "Antisocial Personality Disorder"
STRESS_DISORDER = "Stress Disorder"
SUBSTANCE_ABUSE = "Substance Abuse"
SUBSTANCE_PSYCHOSIS = "Substance-Induced Psychotic Disorder"
TIC_DISORDER = "Tic Disorder"

SYNONYM_MAP = {
    "acute stress disease": STRESS_DISORDER,
    "alcohol use disease": SUBSTANCE_ABUSE,
    "amnestic disease": AMNESTIC_DISORDER,
    "anaemia of chronic disease": ANEMIA,
    "anaesthetic": ANAESTHESIA,
    "anaesthesia": ANAESTHESIA,
    "antisocial personality disease": SOCIOPATHY,
    "articulation disease": ARTICULATION_DISORDER,
    "autism spectrum disease": AUTISM,
    "autism disease": AUTISM,
    "autistic disease": AUTISM,
    "aspergers disease": ASPERGERS,
    "asperger syndrome": ASPERGERS,
    "aspergers syndrome": ASPERGERS,
    "asperger\\'s disease": ASPERGERS,
    "attention deficit hyperactivity disease": ADHD,
    "attention deficit/hyperactivity disease": ADHD,
    "attention deficit disorder with hyperactivity": ADHD,
    "anxiety": ANXIETY,
    "anxiety disease": GENERALIZED_ANXIETY_DISORDER,
    "bipolar disease": BIPOLAR,
    "bipolar i disease": BDI,
    "bipolar ii disease": BDII,
    "body dysmorphic disease": BODY_DYSMORPHIA,
    "borderline personality disease": BPD,
    "celiac disease": CELIAC,
    "coeliac disease": CELIAC,
    "combat disease": PTSD,  # ??
    "communication disease": "communication disorder",
    "cyclothymic disease": CYCLOTHYMIA,
    "generalised anxiety disease": GENERALIZED_ANXIETY_DISORDER,
    "generalized anxiety disease": GENERALIZED_ANXIETY_DISORDER,
    "delusional disease": DELUSION,
    "depressive disease": DEPRESSION,
    "depressed mood": DEPRESSION,
    "depressive symptom": DEPRESSION,
    "depressive": DEPRESSION,
    "dependence": SUBSTANCE_ABUSE,
    "diarrhoea": DIARRHEA,
    "dysthymic disease": DYSTHYMIA,
    "dissociative identity disease": DISSASOCIATIVE_IDENTITY_DISORDER,
    "dissociative disease": DISSASOCIATIVE_IDENTITY_DISORDER,
    "drug dependence": SUBSTANCE_ABUSE,
    "eating disease": EATING_DISORDER,
    "effect on cardiovascular disease": "Cardiovascular disease",
    "female reproductive system disease": REPRODUCTIVE_DISORDER,
    "gastrooesophageal reflux disease": GERD,
    "haemorrhagic disease": "Hemorrhagic disease",
    "haematological disease": HEMATOLOGICAL_DISORDER,
    "haemochromatosis": "Hemochromatosis",
    "hypersensitivity reaction disease": HYPERSENSITIVITY,
    "hypomania": BDII,
    "intermittent explosive disease": INTERMITTENT_EXPLOSIVE_DISORDER,
    "ischaemic disease": ISCHEMIA,
    "ischaemia": ISCHEMIA,
    "manic and bipolar mood disorders and disturbance": BIPOLAR,
    "major depression": DEPRESSION,
    "major depressive disease": DEPRESSION,
    "mental disease": MENTAL_DISORDER,
    "mood disease": MOOD_DISRDER,
    "middle ear disease": "Middle ear disorder",
    "migraine disorders": MIGRAINE,
    "motor neurone disease": MOTOR_NEURON_DISEASE,
    "multiple personality disease": DISSASOCIATIVE_IDENTITY_DISORDER,
    "movement disease": "movement disorder",
    "narcissistic personality disease": NPD,
    "neurotic disease": NEUROSIS,
    "neurotic disorder": NEUROSIS,
    "neurodevelopmental disease": NEURODEVELOPMENTAL_DISORDER,
    "obsessive-compulsive disease": OCD,
    "pain disease": PAIN_DISORDER,
    "panic disease": PANIC,
    "panic attacks and disease": PANIC,
    "pathogenic pain disease": PATHOGENIC_PAIN,
    "penicillins": "penicillin",
    "pervasive developmental disease": DEVELOPMENTAL_DISORDER,
    "personality disease": PERSONALITY_DISORDER,
    "photosensitivity disease": PHOTOSENSITIVITY,
    "phobic disease": PHOBIA,
    "post-traumatic stress disease": PTSD,
    "premenstrual dysphoric disease": PDD,
    "psychosexual disease": SEXUAL_DISORDER,
    "psychotic disease": PSYCHOTIC_DISORDER,
    "psychiatric disease": PSYCHIATRIC_DISORDER,
    "oppositional defiant disease": OPPOSITIONAL_DEFIANT_DISORDER,
    "reading disease": "Reading disorder",
    "schizoaffective disease": SCHIZOAFFECTIVE_DISORDER,
    "schizophreniform disease": SCHIZOPHRENIFORM_DISORDER,
    "brief psychotic disease": BRIEF_PSYCHOTIC_DISORDER,
    "schizophrenia and other psychotic disease": PSYCHOTIC_DISORDER,
    "seasonal affective disease": SAD,
    "sexual arousal disease": SEXUAL_DISORDER,
    "sexual desire disease": SEXUAL_DISORDER,
    "sleep disease": SLEEP_DISORDER,
    "shared psychotic disease": PSYCHOTIC_DISORDER,
    "sleep apnea syndrome": SLEEP_APNEA,
    "sleep apnoea syndrome": SLEEP_APNEA,
    "somatoform disease": "Somatoform disorder",
    "specific developmental disease": DEVELOPMENTAL_DISORDER,
    "stress disease": STRESS_DISORDER,
    "substance related disease": SUBSTANCE_ABUSE,
    "substance-related disease": SUBSTANCE_ABUSE,
    "substance-induced psychotic disease": SUBSTANCE_PSYCHOSIS,
    "tic disease": TIC_DISORDER,
    "transient tic disease": TIC_DISORDER,
}
