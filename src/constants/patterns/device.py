HIGH_LIKELIHOOD_DEVICES = [
    ".*wearable.*",
    "(?:.* )?video.*",
    ".*electronic.*",
    "(?:.* )?dressing",
    "(?:.* )?cannulas?(?: |-|$).*",
    "(?:.* )?(?:back)?rest",
    "(?:.*[ -]?)?inject(?:or)?",  # autoinjector
    "(?:.* )?packag(?:ing|ed)(?: .*)?",  # packaging
    ".*needles?(?: |-|$).*",
    "(?:.* )?bolus",
    ".*mattress.*",
    ".*dental floss.*",
    ".*assembly",
    "(?:.* )?retainer",
    "(?:.* )?infusion pump.*",
    "(?:.* )?pacemaker.*",
    "(?:.* )?scales?(?: |-|$).*",
    "(?:.* )?glass(?:es)?(?: |-|$).*",
    "(?:.* )?nanotube",
    "(?:.* )?rubbers?(?: |-|$).*",
    "(?:.* )?tanks?(?: |-|$).*",
    ".*computer.*",
    ".*fuel cell.*",
    "(?:.* )?latch",
    "(?:.* )?manifold.*",
    "(?:.* )?headgear.*",
    "(?:.* )?clip",
    "(?:.* )?lens",
    "(?:.* )?pulley.*",
    "(?:.* )?belt",
    "(?:.* )?pivot",
    "(?:.* )?mask",
    "(?:.* )?board",
    "(?:.* )?bridge",
    "(?:.* )?cuff",
    "(?:.* )?pouch",
    "(?:.* )?container",
    "(?:.* )?receptacle",
    "(?:.* )?conductor.*",
    "(?:.* )?connector",
    "(?:.* )?effector.*",
    "(?:.* )?tape",
    "(?:.* )?inlet",
    "(?:.* )?outlet",
    "(?:.* )?strip",  # e.g. test strip
    "(?:.* )?solid state",
    "(?:.*)?wire",  # e.g. guidewire
    "(?:.* )?bed",
    "(?:.* )?switch",  # could potentially be biological
    "(?:.* )?prosthetic.*",
    "(?:.* )?equipment",
    "(?:.* )?generator",
    "(?:.* )?cartridge",
    # "(?:.* )?(?:micro)?channel", # icon channel
    "(?:impression|packaging|building) material",
    "(?:.* )?light[ -]?emitt(?:er|ing)s?.*",
    "(?:.* )?cathode.*",
    "(?:.* )?delivery system.*",
    "(?:.* )?dielectric",
    "(?:.* )?roller",
    "(?:.* )?mandrel.*",
    "(?:.* )?stylet",
    "(?:.* )?coupling.*",
    "(?:.* )?attachment",
    "(?:.* )?shaft",
    "(?:.* )?aperture.*",
    "(?:.* )?(?:bio)?sensor",
    "(?:.* )?conduits?(?: |-|$).*",
    ".*scope",
    ".*module",
    "(?:.* )?article",
    "(?:.* )?nozzle.*",
    "(?:.* )?plastics?(?: |-|$).*",
    "(?:.* )?holder",
    "(?:.* )?flanges?(?: |-|$).*",
    ".*circuit.*",
    "(?:.* )?liner?",
    "(?:.* )?pole",
    "(?:.* )?paper",
    "(?:.* )?light",  # light chain? (bio)
    "(?:.* )?solar cell.*",
    "(?:.* )?ground",
    "(?:.* )?waveform",
    "(?:.* )?tool",
    "(?:.* )?sequencer",
    "(?:.* )?centrifuge",
    "(?:.* )?surfaces?(?: |-|$).*",
    "(?:.* )?recording medium(?: |-|$).*",
    "(?:.* )?field",
    "(?:.* )?garment(?: |-|$).*",
    "(?:.* )?mou?ld(?:ed|ing)?(?: .*)?",  # molding, moulded, molded product
    "(?:.* )?napkins?(?: |-|$).*",
    "(?:.* )?anvil.*",
    "(?:.* )?wheelchair.*",
    "(?:.* )?walls?(?: |-|$).*",
    "(?:.* )?wheel.*",
    "(?:.* )?manipulators?(?: |-|$).*",
    ".*(?: |^|-)gasket.*",
    ".*(?: |^|-)ratchet.*",
    ".*(?: |^|-)syringe.*",
    ".*(?: |^|-)canister.*",
    "(?:.* )?slides?(?: |-|$).*",
    "(?:.* )?tethers?(?: |-|$).*",
    ".*meter(?: |-|$).*",
    "(?:.* )?diaper.*",
    "(?:.* )?coils?(?: |-|$).*",
    "(?:.* )?apparatuses?.*",
    "(?:.* )?waveguide.*",
    "(?:.* )?implants?(?: |-|$).*",
    ".*(?: |^|-)rods?(?: |-|$).*",
    ".*(?: |^|-)conductive",
    "(?:.* )?cushion.*",
    ".*(?: |^|-)trocar.*",
    ".*(?: |^|-)liquid crystal.*",
    "(?:.* )?prosthesis.*",
    "(?:.* )?catheter.*",
    # "(?:.* )?film",
    ".*(?: |^|-)hinge",
    ".*(?: |^|-)instrument",
    "(?:.* )?vessel",  # ??
    ".*(?: |^|-)devices?.*",
    "(?:.* )?motor",  # motor disorders
    ".*(?: |^|-)electrode.*",
    "(?:.* )?camera.*",
    ".*(?: |^|-)mouthpiece.*",
    ".*(?: |^|-)transducer.*",
    "(?:.* )?toothbrush.*",
    "(?:.* )?suture(?: |-|$).*",
    "(?:.* )?stent(?: |-|$).*",
    "(?:.* )?plate(?: |-|$).*",
    "(?:.* )?dispenser.*",
    "(?:.* )?tip(?: |-|$).*",
    "(?:.* )?probe(?: |-|$).*",
    "(?:.* )?wafer(?: |-|$).*",
    "(?:.* )?fastener.*",
    "(?:.* )?(?:bike|bicycle)(?: |-|$).*",
    "(?:.* )?diaphrag?m.*",
    "(?:.* )?plunger.*",
    "(?:.* )?piston.*",
    "(?:.* )?balloon.*",
    # "(?:.* )?linkage",  # ?? bio
    "(?:.* )?(?:bio)?reactor",
    ".*piezoelectric.*",
    ".*ultrasonic.*",
    ".*splitter",
    "(?:.* )?spacer(?: |-|$).*",
    "(?:.* )?strut(?: |-|$).*",
    "(?:.* )?capacitor.*",
    "(?:.* )?reservoir.*",
    "(?:.* )?housing.*",
    "(?:.* )?tub(?:e|ing)",
    "(?:.* )?lancet(?: |-|$).*",
    "(?:.* )?appliance",
    "(?:.* )?pump",
    "(?:.* )?cutter(?: |-|$).*",
    "(?:.* )?compressor(?: |-|$).*",
    "(?:.* )?forcep(?: |-|$).*",
    "(?:.* )?batter(?:y|ie)",
    "(?:.* )?blade",  # blade
    "(?:.* )?machine",
    "(?:.* )?diode",
    "(?:.* )?accelerator",
    "(?:.* )?indicator",
    "(?:.* )?pump",
    "(?:.* )?chamber",
    "(?:.* )?clamp",
    "(?:.* )?array(?: |-|$).*",
    "(?:.* )?compartment",
    "(?:.* )?stapler?(?: |-|$).*",
    "(?:.* )?radiator(?: |-|$).*",
    "(?:.* )?actuator(?: |-|$).*",
    "(?:.* )?engine(?: |-|$).*",
    "(?:fuel|electrochemical) cell(?: |-|$).*",
    "(?:.* )?screw(?: |-|$).*",
    "(?:.* )?monitor",
    ".*electrical stimulater(?: |-|$).*",
    ".*grapher",
    ".*software.*",  # in this context, SAMD
    ".*clinical decision support.*",
    "cdss",
    "robot.*",
    "(?:.* )?fabric",
    "(?:.* )?tampon(?: |-|$).*",
    "(?:.* )?(?:top)?sheet",
    "(?:.*[ -])?scale",
    ".*microprocessor.*",
]
# for use in patents; might need adjustment to be more generally applicable.
DEVICE_RES = [
    *HIGH_LIKELIHOOD_DEVICES,
    "(?:.* )?fibers?(?: |-|$).*",  # TODO could be bio
    "(?:.* )?head",
    "(?:.* )?source",  # TODO
    "(?:.* )?windows?(?: |-|$).*",
    "(?:.* )?body",
    "(?:.* )?sheath",
    "(?:.* )?elements?(?: |-|$).*",  # e.g. locking element # TODO probably excessive
    "(?:.* )?table",
    "(?:.* )?displays?(?: |-|$).*",
    "(?:.* )?port",
    "(?:.* )?seperator",
    "(?:.* )?pads?(?: |-|$).*",
    "(?:.* )?wave",
    "(?:.* )?current",
    "(?:.* )?article",
    "(?:.* )?valve",
    "(?:.* )?bladder",
    "(?:.* )?sponges?(?: |-|$).*",
    "(?:.* )?textile.*",
    "(?:.* )?lead",
    "(?:.* )?block",  # TODO: procedure?
    "(?:.* )?mobile(?: |-|$).*",
    "core[ -]?shell",  # what is this?
    "stop(?:per)?",
    ".*mechanical.*",
    ".*transceiver",
    "(?:.* )?impeller.*",
    "(?:.* )?transmitter.*",  # sexually transmitted disease
    "(?:.* )?slider(?: |-|$).*",
    "(?:.* )?abutment(?: |-|$).*",
    "fastening mean",
    ".*handpiece.*",
    "/*centrifugal force.*",
    ".*sealant.*",
    ".*microelectronic.*",
]
