DROP TABLE IF EXISTS MRCOLS;
CREATE TABLE MRCOLS (
    COL	varchar(40),
    DES	varchar(200),
    REF	varchar(40),
    MIN	int,
    AV	numeric(5,2),
    MAX	int,
    FIL	varchar(50),
    DTY	varchar(40)
);

\copy MRCOLS FROM 'MRCOLS.RRF' WITH DELIMITER '|';


DROP TABLE IF EXISTS MRCONSO;
CREATE TABLE MRCONSO (
    CUI	char(8) NOT NULL,
    LAT	char(3) NOT NULL,
    TS	char(1) NOT NULL,
    LUI	varchar(10) NOT NULL,
    STT	varchar(3) NOT NULL,
    SUI	varchar(10) NOT NULL,
    ISPREF	char(1) NOT NULL,
    AUI	varchar(9) NOT NULL,
    SAUI	varchar(50),
    SCUI	varchar(100),
    SDUI	varchar(100),
    SAB	varchar(40) NOT NULL,
    TTY	varchar(40) NOT NULL,
    CODE	varchar(100) NOT NULL,
    STR	text NOT NULL,
    SRL	int NOT NULL,
    SUPPRESS	char(1) NOT NULL,
    CVF	int
);

\copy MRCONSO FROM 'MRCONSO.RRF' WITH (DELIMITER '|', FORCE_NULL (CVF), FORMAT CSV);

DROP TABLE IF EXISTS MRCUI;
CREATE TABLE MRCUI (
    CUI1	char(8) NOT NULL,
    VER	varchar(10) NOT NULL,
    REL	varchar(4) NOT NULL,
    RELA	varchar(100),
    MAPREASON	text,
    CUI2	char(8),
    MAPIN	char(1)
);

\copy MRCUI FROM 'MRCUI.RRF' WITH DELIMITER '|';

DROP TABLE IF EXISTS MRCXT;
CREATE TABLE MRCXT (
    CUI	char(8),
    SUI	varchar(10),
    AUI	varchar(9),
    SAB	varchar(40),
    CODE	varchar(100),
    CXN	int,
    CXL	char(3),
    MRCXTRANK	int,
    CXS	text,
    CUI2	char(8),
    AUI2	varchar(9),
    HCD	varchar(100),
    RELA	varchar(100),
    XC	varchar(1),
    CVF	int
);

\copy MRCXT FROM 'MRCXT.RRF' WITH (DELIMITER '|', FORCE_NULL (CVF), FORMAT CSV);

DROP TABLE IF EXISTS MRDEF;
CREATE TABLE MRDEF (
    CUI	char(8) NOT NULL,
    AUI	varchar(9) NOT NULL,
    ATUI	varchar(11) NOT NULL,
    SATUI	varchar(50),
    SAB	varchar(40) NOT NULL,
    DEF	text NOT NULL,
    SUPPRESS	char(1) NOT NULL,
    CVF	int
);

\copy MRDEF FROM 'MRDEF.RRF' WITH (DELIMITER '|', FORCE_NULL (CVF), FORMAT CSV);

DROP TABLE IF EXISTS MRDOC;
CREATE TABLE MRDOC (
    DOCKEY	varchar(50) NOT NULL,
    VALUE	varchar(200),
    TYPE	varchar(50) NOT NULL,
    EXPL	text
);

\copy MRDOC FROM 'MRDOC.RRF' WITH DELIMITER '|';

DROP TABLE IF EXISTS MRFILES;
CREATE TABLE MRFILES (
    FIL	varchar(50),
    DES	varchar(200),
    FMT	text,
    CLS	int,
    RWS	int,
    BTS	bigint
);

\copy MRFILES FROM 'MRFILES.RRF' WITH DELIMITER '|';

DROP TABLE IF EXISTS MRHIER;
CREATE TABLE MRHIER (
    CUI	char(8) NOT NULL,
    AUI	varchar(9) NOT NULL,
    CXN	int NOT NULL,
    PAUI	varchar(10),
    SAB	varchar(40) NOT NULL,
    RELA	varchar(100),
    PTR	text,
    HCD	varchar(100),
    CVF	int
);

\copy MRHIER FROM 'MRHIER.RRF' WITH (DELIMITER '|', FORCE_NULL (CVF), FORMAT CSV);

DROP TABLE IF EXISTS MRHIST;
CREATE TABLE MRHIST (
    CUI	char(8),
    SOURCEUI	varchar(100),
    SAB	varchar(40),
    SVER	varchar(40),
    CHANGETYPE	text,
    CHANGEKEY	text,
    CHANGEVAL	text,
    REASON	text,
    CVF	int
);

\copy MRHIST FROM 'MRHIST.RRF' WITH DELIMITER '|';


DROP TABLE IF EXISTS MRMAP;
CREATE TABLE MRMAP (
    MAPSETCUI	char(8) NOT NULL,
    MAPSETSAB	varchar(40) NOT NULL,
    MAPSUBSETID	varchar(10),
    MAPRANK	int,
    MAPID	varchar(50) NOT NULL,
    MAPSID	varchar(50),
    FROMID	varchar(50) NOT NULL,
    FROMSID	varchar(50),
    FROMEXPR	text NOT NULL,
    FROMTYPE	varchar(50) NOT NULL,
    FROMRULE	text,
    FROMRES	text,
    REL	varchar(4) NOT NULL,
    RELA	varchar(100),
    TOID	varchar(50),
    TOSID	varchar(50),
    TOEXPR	text,
    TOTYPE	varchar(50),
    TORULE	text,
    TORES	text,
    MAPRULE	text,
    MAPRES	text,
    MAPTYPE	varchar(50),
    MAPATN	varchar(100),
    MAPATV	text,
    CVF	int
);

\copy MRMAP FROM 'MRMAP.RRF' WITH (DELIMITER '|', FORCE_NULL (CVF), FORMAT CSV);


DROP TABLE IF EXISTS MRRANK;
CREATE TABLE MRRANK (
    MRRANK_RANK	int NOT NULL,
    SAB	varchar(40) NOT NULL,
    TTY	varchar(40) NOT NULL,
    SUPPRESS	char(1) NOT NULL
);

\copy MRRANK FROM 'MRRANK.RRF' WITH DELIMITER '|';


DROP TABLE IF EXISTS MRREL;
CREATE TABLE MRREL (
    CUI1	char(8) NOT NULL,
    AUI1	varchar(9),
    STYPE1	varchar(50) NOT NULL,
    REL	varchar(4) NOT NULL,
    CUI2	char(8) NOT NULL,
    AUI2	varchar(9),
    STYPE2	varchar(50) NOT NULL,
    RELA	varchar(100),
    RUI	varchar(10) NOT NULL,
    SRUI	varchar(50),
    SAB	varchar(40) NOT NULL,
    SL	varchar(40) NOT NULL,
    RG	varchar(10),
    DIR	varchar(1),
    SUPPRESS	char(1) NOT NULL
);

\copy MRREL FROM 'MRREL.RRF' WITH (DELIMITER '|');




DROP TABLE IF EXISTS MRSAB;
CREATE TABLE MRSAB (
    VCUI	char(8),
    RCUI	char(8),
    VSAB	varchar(40) NOT NULL,
    RSAB	varchar(40) NOT NULL,
    SON	text NOT NULL,
    SF	varchar(40) NOT NULL,
    SVER	varchar(40),
    VSTART	char(8),
    VEND	char(8),
    IMETA	varchar(10) NOT NULL,
    RMETA	varchar(10),
    SLC	text,
    SCC	text,
    SRL	int NOT NULL,
    TFR	int,
    CFR	int,
    CXTY	varchar(50),
    TTYL	varchar(400),
    ATNL	text,
    LAT	char(3),
    CENC	varchar(40) NOT NULL,
    CURVER	char(1) NOT NULL,
    SABIN	char(1) NOT NULL,
    SSN	text NOT NULL,
    SCIT	text NOT NULL
);

\copy MRSAB FROM 'MRSAB.RRF' WITH DELIMITER '|';



DROP TABLE IF EXISTS MRSAT;
CREATE TABLE MRSAT (
    CUI	char(8) NOT NULL,
    LUI	varchar(10),
    SUI	varchar(10),
    METAUI	varchar(100),
    STYPE	varchar(50) NOT NULL,
    CODE	varchar(100),
    ATUI	varchar(11) NOT NULL,
    SATUI	varchar(50),
    ATN	varchar(100) NOT NULL,
    SAB	varchar(40) NOT NULL,
    ATV	text,
    SUPPRESS	char(1) NOT NULL,
    CVF	int
);

\copy MRSAT FROM 'MRSAT.RRF' WITH (DELIMITER '|', FORCE_NULL (CVF), FORMAT CSV);



DROP TABLE IF EXISTS MRSMAP;
CREATE TABLE MRSMAP (
    MAPSETCUI	char(8) NOT NULL,
    MAPSETSAB	varchar(40) NOT NULL,
    MAPID	varchar(50) NOT NULL,
    MAPSID	varchar(50),
    FROMEXPR	text NOT NULL,
    FROMTYPE	varchar(50) NOT NULL,
    REL	varchar(4) NOT NULL,
    RELA	varchar(100),
    TOEXPR	text,
    TOTYPE	varchar(50),
    CVF	int
);

\copy MRSMAP FROM 'MRSMAP.RRF' WITH (DELIMITER '|', FORCE_NULL (CVF), FORMAT CSV);


DROP TABLE IF EXISTS MRSTY;
CREATE TABLE MRSTY (
    CUI	char(8) NOT NULL,
    TUI	char(4) NOT NULL,
    STN	varchar(100) NOT NULL,
    STY	varchar(50) NOT NULL,
    ATUI	varchar(11) NOT NULL,
    CVF	int
);

\copy MRSTY FROM 'MRSTY.RRF' WITH (DELIMITER '|', FORCE_NULL (CVF), FORMAT CSV);


DROP TABLE IF EXISTS MRXNS_ENG;
CREATE TABLE MRXNS_ENG (
    LAT	char(3) NOT NULL,
    NSTR	text NOT NULL,
    CUI	char(8) NOT NULL,
    LUI	varchar(10) NOT NULL,
    SUI	varchar(10) NOT NULL
);

\copy MRXNS_ENG FROM 'MRXNS_ENG.RRF' WITH DELIMITER '|';


DROP TABLE IF EXISTS MRXNW_ENG;
CREATE TABLE MRXNW_ENG (
    LAT	char(3) NOT NULL,
    NWD	varchar(100) NOT NULL,
    CUI	char(8) NOT NULL,
    LUI	varchar(10) NOT NULL,
    SUI	varchar(10) NOT NULL
);

\copy MRXNW_ENG FROM 'MRXNW_ENG.RRF' WITH DELIMITER '|';


DROP TABLE IF EXISTS MRAUI;
CREATE TABLE MRAUI (
    AUI1	varchar(9) NOT NULL,
    CUI1	char(8) NOT NULL,
    VER	varchar(10) NOT NULL,
    REL	varchar(4),
    RELA	varchar(100),
    MAPREASON	text NOT NULL,
    AUI2	varchar(9) NOT NULL,
    CUI2	char(8) NOT NULL,
    MAPIN	char(1) NOT NULL
);

\copy MRAUI FROM 'MRAUI.RRF' WITH DELIMITER '|';


DROP TABLE IF EXISTS MRXW_ENG;
CREATE TABLE MRXW_ENG (
    LAT	char(3) NOT NULL,
    WD	varchar(200) NOT NULL,
    CUI	char(8) NOT NULL,
    LUI	varchar(10) NOT NULL,
    SUI	varchar(10) NOT NULL
);

\copy MRXW_ENG FROM 'MRXW_ENG.RRF' WITH DELIMITER '|';


DROP TABLE IF EXISTS AMBIGSUI;
CREATE TABLE AMBIGSUI (
    SUI	varchar(10) NOT NULL,
    CUI	char(8) NOT NULL
);


\copy AMBIGSUI FROM 'AMBIGSUI.RRF' WITH DELIMITER '|';


DROP TABLE IF EXISTS AMBIGLUI;
CREATE TABLE AMBIGLUI (
    LUI	varchar(10) NOT NULL,
    CUI	char(8) NOT NULL
);

\copy AMBIGLUI FROM 'AMBIGLUI.RRF' WITH DELIMITER '|';


DROP TABLE IF EXISTS DELETEDCUI;
CREATE TABLE DELETEDCUI (
    PCUI	char(8) NOT NULL,
    PSTR	text NOT NULL
);


\copy DELETEDCUI FROM 'CHANGE/DELETEDCUI.RRF' WITH DELIMITER '|';



DROP TABLE IF EXISTS DELETEDLUI;
CREATE TABLE DELETEDLUI (
    PLUI	varchar(10) NOT NULL,
    PSTR	text NOT NULL
);

\copy DELETEDLUI FROM 'CHANGE/DELETEDLUI.RRF' WITH DELIMITER '|';



DROP TABLE IF EXISTS DELETEDSUI;
CREATE TABLE DELETEDSUI (
    PSUI	varchar(10) NOT NULL,
    LAT	char(3) NOT NULL,
    PSTR	text NOT NULL
);

\copy DELETEDSUI FROM 'CHANGE/DELETEDSUI.RRF' WITH DELIMITER '|';


DROP TABLE IF EXISTS MERGEDCUI;
CREATE TABLE MERGEDCUI (
    PCUI	char(8) NOT NULL,
    CUI	char(8) NOT NULL
);

\copy MERGEDCUI FROM 'CHANGE/MERGEDCUI.RRF' WITH DELIMITER '|';

DROP TABLE IF EXISTS MERGEDLUI;
CREATE TABLE MERGEDLUI (
    PLUI	varchar(10),
    LUI	varchar(10)
);

\copy MERGEDLUI FROM 'CHANGE/MERGEDLUI.RRF' WITH DELIMITER '|';
