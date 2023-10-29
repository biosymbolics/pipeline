HIGH_LIKELIHOOD_DEVICES = [
    ".*wearable.*",
    "(?:.* )?video",
    ".*electronic.*",
    "(?:.* )?dressing",
    "(?:.* )?cannula",
    "(?:.* )?(?:back)?rest",
    "(?:.*[ -]?)?inject(?:or)?",  # autoinjector
    "(?:.* )?packag(?:ing|ed)(?: .*)?",  # packaging
    ".*needle",
    "(?:.* )?bolus",
    ".*mattress.*",
    ".*dental floss",
    ".*assembly",
    "(?:.* )?retainer",
    "(?:.* )?infusion pump",
    "(?:.* )?pacemaker",
    "(?:.* )?glass",
    "(?:.* )?nanotube",
    "(?:.* )?rubber",
    "(?:.* )?tank",
    ".*computer.*",
    ".*fuel cell.*",
    "(?:.* )?latch",
    "(?:.* )?manifold",
    "(?:.* )?headgear",
    "(?:.* )?clip",
    "(?:.* )?lens",
    "(?:.* )?pulley",
    "(?:.* )?belt",
    "(?:.* )?pivot",
    "(?:.* )?mask",
    "(?:.* )?board",
    "(?:.* )?bridge",
    "(?:.* )?cuff",
    "(?:.* )?pouch",
    "(?:.* )?container",
    "(?:.* )?receptacle",
    "(?:.* )?conductor",
    "(?:.* )?connector",
    "(?:.* )?effector",
    "(?:.* )?tape",
    "(?:.* )?inlet",
    "(?:.* )?outlet",
    "(?:.* )?strip",  # e.g. test strip
    "(?:.* )?solid state",
    "(?:.*)?wire",  # e.g. guidewire
    "(?:.* )?bed",
    "(?:.* )?switch",  # could potentially be biological
    "(?:.* )?prosthetic",
    "(?:.* )?equipment",
    "(?:.* )?generator",
    "(?:.* )?cartridge",
    "(?:.* )?(?:micro)?channel",
    "impression material",
    "building material",
    "(?:.* )?light[ -]?emitt(?:er|ing)s?.*",
    "(?:.* )?cathode",
    "(?:.* )?delivery system",
    "(?:.* )?dielectric",
    "(?:.* )?roller",
    "(?:.* )?mandrel",
    "(?:.* )?stylet",
    "(?:.* )?coupling",
    "(?:.* )?attachment",
    "(?:.* )?shaft",
    "(?:.* )?aperture",
    "(?:.* )?(?:bio)?sensor",
    "(?:.* )?conduit",
    ".*scope",
    ".*module",
    "(?:.* )?article",
    "(?:.* )?nozzle",
    "(?:.* )?plastic",
    "(?:.* )?holder",
    "(?:.* )?flange",
    "(?:.* )?circuit",
    "(?:.* )?liner",
    "(?:.* )?paper",
    "(?:.* )?light",  # light chain? (bio)
    "(?:.* )?solar cell.*",
    "(?:.* )?ground",
    "(?:.* )?waveform",
    "(?:.* )?tool",
    "(?:.* )?sequencer",
    "(?:.* )?centrifuge",
    "(?:.* )?surface",
    "(?:.* )?recording medium",
    "(?:.* )?field",
    "(?:.* )?garment",
    "(?:.* )?mou?ld(?:ed|ing)?(?: .*)?",  # molding, moulded, molded product
    "(?:.* )?napkin",
    "(?:.* )?anvil",
    "(?:.* )?wheelchair",
    "(?:.* )?wall",
    "(?:.* )?wheel",
    "(?:.* )?manipulator",
    "(?:.* )?gasket",
    "(?:.* )?ratchet",
    "(?:.* )?syringe(?: .*)?",
    "(?:.* )?canist",
    "(?:.* )?slide",
    "(?:.* )?tether",
    ".*meter",
    "(?:.* )?diaper",
    "(?:.* )?coil",
    "(?:.* )?apparatuse?",
    "(?:.* )?waveguide",
    "(?:.* )?implant",
    "(?:.* )?rod",
    "(?:.* )?conductive",
    "(?:.* )?cushion",
    "(?:.* )?trocar",
    "(?:.* )?liquid crystal(?: .*)?",
    "(?:.* )?prosthesis",
    "(?:.* )?catheter.*",
    "(?:.* )?film",
    "(?:.* )?hinge",
    "(?:.* )?instrument",
    "(?:.* )?vessel",
    "(?:.* )?device(?: .*)?",
    "(?:.* )?motor",
    "(?:.* )?electrode",
    "(?:.* )?camera",
    "(?:.* )?mouthpiece",
    "(?:.* )?transducer",
    "(?:.* )?toothbrush",
    "(?:.* )?suture",
    "(?:.* )?stent",
    "(?:.* )?plate",
    "(?:.* )?dispenser",
    "(?:.* )?tip",
    "(?:.* )?probe",
    "(?:.* )?wafer",
    "(?:.* )?fastener",
    "(?:.* )?(?:bike|bicycle)",
    "(?:.* )?diaphrag?m",
    "(?:.* )?plunger",
    "(?:.* )?piston",
    "(?:.* )?balloon",
    "(?:.* )?linkage",  # ??
    "(?:.* )?(?:bio)?reactor",
    ".*piezoelectric.*",
    ".*ultrasonic.*",
    "(?:.* )?splitter",
    "(?:.* )?spacer",
    "(?:.* )?strut",
    "(?:.* )?capacitor",
    "(?:.* )?reservoir",
    "(?:.* )?housing",
    "(?:.* )?tube",
    "(?:.* )?lancet",
    "(?:.* )?appliance",
    "(?:.* )?pump",
    "(?:.* )?cutter",
    "(?:.* )?compressor",
    "(?:.* )?forcep",
    "(?:.* )?batter(?:y|ie)",
    "(?:.* )?blade",  # blade
    "(?:.* )?machine",
    "(?:.* )?diode",
    "(?:.* )?accelerator",
    "(?:.* )?indicator",
    "(?:.* )?pump",
    "(?:.* )?chamber",
    "(?:.* )?clamp",
    "(?:.* )?compartment",
    "(?:.* )?stapler?",
    "(?:.* )?radiator",
    "(?:.* )?actuator",
    "(?:.* )?engine",
    "(?:fuel|electrochemical) cell",
    "(?:.* )?screw",
    "(?:.* )?monitor",
    ".*electrical stimulater",
    ".*grapher",
    ".*software.*",  # in this context, SAMD
    ".*clinical decision support.*",
    "cdss",
    "robot.*",
]
# for use in patents; might need adjustment to be more generally applicable.
DEVICE_RES = [
    *HIGH_LIKELIHOOD_DEVICES,
    "(?:.* )?fiber",  # TODO could be bio
    "(?:.* )?head",
    "(?:.* )?source",  # TODO
    "(?:.* )?window",
    "(?:.* )?body",
    "(?:.* )?sheath",
    "(?:.* )?element",  # e.g. locking element # TODO probably excessive
    "(?:.* )?table",
    "(?:.* )?display",
    "(?:.*[ -])?scale",
    "(?:.* )?port",
    "(?:.* )?seperator",
    "(?:.* )?program",
    "(?:.* )?fabric",
    "(?:.* )?tampon",
    "(?:.* )?(?:top)?sheet",
    "(?:.* )?pad",
    "(?:.* )?wave",
    "(?:.* )?current",
    "(?:.* )?article",
    "(?:.* )?valve",
    "(?:.* )?bladder",
    "(?:.* )?sponge",
    "(?:.* )?textile",
    "(?:.* )?lead",
    "(?:.* )?block",  # TODO: procedure?
    "mobile",
    "core[ -]?shell",  # what is this?
    "stop(?:per)?",
    ".*mechanical.*",
    ".*transceiver",
    "impeller",
    "transmit",
    "slider",
    "abutment",
    "fastening mean",
    "handpiece",
    "reagent kit",
    "centrifugal force",
    "sealant",
    "microelectronic",
]
