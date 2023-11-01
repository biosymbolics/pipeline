
-- This script creates all the UMLS tables (with some errors not important to our current use-case)
DROP TABLE MRCOLS;
CREATE TABLE MRCOLS (
	COL	varchar(20),
	DES	varchar(200),
	REF	varchar(20),
	MIN	int,
	AV	numeric(5,2),
	MAX	int,
	FIL	varchar(50),
	DTY	varchar(20)

);
\copy MRCOLS from 'META/MRCOLS.RRF' with delimiter as '|' null as '';


DROP TABLE MRCONSO;
CREATE TABLE MRCONSO (
	CUI	char(9) NOT NULL,
	LAT	char(3) NOT NULL,
	TS	char(1) NOT NULL,
	LUI	char(9) NOT NULL,
	STT	varchar(3) NOT NULL,
	SUI	char(9) NOT NULL,
	ISPREF	char(1) NOT NULL,
	AUI	varchar(9) NOT NULL,
	SAUI	varchar(50),
	SCUI	varchar(200),
	SDUI	varchar(50),
	SAB	varchar(20) NOT NULL,
	TTY	varchar(20) NOT NULL,
	CODE	varchar(200) NOT NULL,
	STR	text NOT NULL,
	SRL	int NOT NULL,
	SUPPRESS	char(1) NOT NULL,
	CVF	int

);
\copy MRCONSO from 'META/MRCONSO.RRF' with delimiter as '|' null as '';


DROP TABLE MRCUI;
CREATE TABLE MRCUI (
	CUI1	char(8) NOT NULL,
	VER	varchar(10) NOT NULL,
	REL	varchar(4) NOT NULL,
	RELA	varchar(100),
	MAPREASON	text,
	CUI2	char(8),
	MAPIN	char(1)

);
\copy MRCUI from 'META/MRCUI.RRF' with delimiter as '|' null as '';


DROP TABLE MRCXT;
CREATE TABLE MRCXT (
	CUI	char(9),
	SUI	char(9),
	AUI	varchar(9),
	SAB	varchar(20),
	CODE	varchar(50),
	CXN	int,
	CXL	char(3),
	RANK	int,
	CXS	text,
	CUI2	char(8),
	AUI2	varchar(9),
	HCD	varchar(50),
	RELA	varchar(100),
	XC	varchar(1),
	CVF	int

);
\copy MRCXT from 'META/MRCXT.RRF' with delimiter as '|' null as '';


DROP TABLE MRDEF;
CREATE TABLE MRDEF (
	CUI	char(9) NOT NULL,
	AUI	varchar(9) NOT NULL,
	ATUI	varchar(12) NOT NULL,
	SATUI	varchar(50),
	SAB	varchar(20) NOT NULL,
	DEF	text NOT NULL,
	SUPPRESS	char(1) NOT NULL,
	CVF	int

);
\copy MRDEF from 'META/MRDEF.RRF' with delimiter as '|' null as '';


DROP TABLE MRDOC;
CREATE TABLE MRDOC (
	DOCKEY	varchar(50) NOT NULL,
	VALUE	varchar(200),
	TYPE	varchar(50) NOT NULL,
	EXPL	text

);
\copy MRDOC from 'META/MRDOC.RRF' with delimiter as '|' null as '';


DROP TABLE MRFILES;
CREATE TABLE MRFILES (
	FIL	varchar(50),
	DES	varchar(200),
	FMT	text,
	CLS	int,
	RWS	int,
	BTS	bigint

);
\copy MRFILES from 'META/MRFILES.RRF' with delimiter as '|' null as '';


DROP TABLE MRHIER;
CREATE TABLE MRHIER (
	CUI	char(9) NOT NULL,
	AUI	varchar(9) NOT NULL,
	CXN	int NOT NULL,
	PAUI	varchar(9),
	SAB	varchar(20) NOT NULL,
	RELA	varchar(100),
	PTR	text,
	HCD	text
	-- CVF	int

);
\copy MRHIER from 'META/MRHIER.RRF' with delimiter as '|' null as '';


DROP TABLE MRHIST;
CREATE TABLE MRHIST (
	CUI	char(9) NOT NULL,
	SOURCEUI	varchar(50) NOT NULL,
	SAB	varchar(20) NOT NULL,
	SVER	varchar(20) NOT NULL,
	CHANGETYPE	text NOT NULL,
	CHANGEKEY	text NOT NULL,
	CHANGEVAL	text NOT NULL,
	REASON	text,
	CVF	int

);
\copy MRHIST from 'META/MRHIST.RRF' with delimiter as '|' null as '';


DROP TABLE MRMAP;
CREATE TABLE MRMAP (
	MAPSETCUI	char(9),
	MAPSETSAB	varchar(20),
	MAPSUBSETID	varchar(10),
	MAPRANK	int,
	MAPID	varchar(50),
	MAPSID	varchar(50),
	FROMID	varchar(50),
	FROMSID	varchar(50),
	FROMEXPR	text,
	FROMTYPE	varchar(50),
	FROMRULE	text,
	FROMRES	text,
	REL	varchar(4),
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
	MAPATN	varchar(20),
	MAPATV	text,
	CVF	int

);
\copy MRMAP from 'META/MRMAP.RRF' with delimiter as '|' null as '';


DROP TABLE MRRANK;
CREATE TABLE MRRANK (
	RANK	int NOT NULL,
	SAB	varchar(20) NOT NULL,
	TTY	varchar(20) NOT NULL,
	SUPPRESS	char(1) NOT NULL

);
\copy MRRANK from 'META/MRRANK.RRF' with delimiter as '|' null as '';


DROP TABLE MRREL;
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
	SAB	varchar(20) NOT NULL,
	SL	varchar(20) NOT NULL,
	RG	varchar(10),
	DIR	varchar(1),
	SUPPRESS	char(1) NOT NULL
	-- CVF	int

);
\copy MRREL from 'META/MRREL.RRF' with delimiter as '|' null as '';


DROP TABLE MRSAB;
CREATE TABLE MRSAB (
	VCUI	char(9),
	RCUI	char(9),
	VSAB	varchar(100) NOT NULL,
	RSAB	varchar(20) NOT NULL,
	SON	text NOT NULL,
	SF	varchar(20) NOT NULL,
	SVER	varchar(20),
	VSTART	char(10),
	VEND	char(10),
	IMETA	varchar(10) NOT NULL,
	RMETA	varchar(10),
	SLC	text,
	SCC	text,
	SRL	int NOT NULL,
	TFR	int,
	CFR	int,
	CXTY	varchar(50),
	TTYL	varchar(200),
	ATNL	text,
	LAT	char(3),
	CENC	varchar(20) NOT NULL,
	CURVER	char(1) NOT NULL,
	SABIN	char(1) NOT NULL,
	SSN	text NOT NULL,
	SCIT	text NOT NULL

);
\copy MRSAB from 'META/MRSAB.RRF' with delimiter as '|' null as '';


DROP TABLE MRSAT;
CREATE TABLE MRSAT (
	CUI	char(9) NOT NULL,
	LUI	char(9),
	SUI	char(9),
	METAUI	varchar(50),
	STYPE	varchar(50) NOT NULL,
	CODE	varchar(50),
	ATUI	varchar(12) NOT NULL,
	SATUI	varchar(50),
	ATN	varchar(50) NOT NULL,
	SAB	varchar(20) NOT NULL,
	ATV	text,
	SUPPRESS	char(1) NOT NULL,
	CVF	int

);
\copy MRSAT from 'META/MRSAT.RRF' with delimiter as '|' null as '';


DROP TABLE MRSMAP;
CREATE TABLE MRSMAP (
	MAPSETCUI	char(9),
	MAPSETSAB	varchar(20),
	MAPID	varchar(50),
	MAPSID	varchar(50),
	FROMEXPR	text,
	FROMTYPE	varchar(50),
	REL	varchar(4),
	RELA	varchar(100),
	TOEXPR	text,
	TOTYPE	varchar(50),
	CVF	int

);
\copy MRSMAP from 'META/MRSMAP.RRF' with delimiter as '|' null as '';


DROP TABLE MRSTY;
CREATE TABLE MRSTY (
	CUI	char(9) NOT NULL,
	TUI	char(4) NOT NULL,
	STN	varchar(100) NOT NULL,
	STY	varchar(50) NOT NULL,
	ATUI	varchar(12) NOT NULL,
	CVF	int

);
\copy MRSTY from 'META/MRSTY.RRF' with delimiter as '|' null as '';


DROP TABLE MRXNS_ENG;
CREATE TABLE MRXNS_ENG (
	LAT	char(3) NOT NULL,
	NSTR	text NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXNS_ENG from 'META/MRXNS_ENG.RRF' with delimiter as '|' null as '';


DROP TABLE MRXNW_ENG;
CREATE TABLE MRXNW_ENG (
	LAT	char(3) NOT NULL,
	NWD	varchar(100) NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXNW_ENG from 'META/MRXNW_ENG.RRF' with delimiter as '|' null as '';


DROP TABLE MRAUI;
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
\copy MRAUI from 'META/MRAUI.RRF' with delimiter as '|' null as '';


DROP TABLE MRXW_BAQ;
CREATE TABLE MRXW_BAQ (
	LAT	char(3) NOT NULL,
	WD	varchar(100) NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXW_BAQ from 'META/MRXW_BAQ.RRF' with delimiter as '|' null as '';


DROP TABLE MRXW_CZE;
CREATE TABLE MRXW_CZE (
	LAT	char(3) NOT NULL,
	WD	varchar(100) NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXW_CZE from 'META/MRXW_CZE.RRF' with delimiter as '|' null as '';


DROP TABLE MRXW_DAN;
CREATE TABLE MRXW_DAN (
	LAT	char(3) NOT NULL,
	WD	varchar(100) NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXW_DAN from 'META/MRXW_DAN.RRF' with delimiter as '|' null as '';


DROP TABLE MRXW_DUT;
CREATE TABLE MRXW_DUT (
	LAT	char(3) NOT NULL,
	WD	varchar(100) NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXW_DUT from 'META/MRXW_DUT.RRF' with delimiter as '|' null as '';


DROP TABLE MRXW_ENG;
CREATE TABLE MRXW_ENG (
	LAT	char(3) NOT NULL,
	WD	varchar(100) NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXW_ENG from 'META/MRXW_ENG.RRF' with delimiter as '|' null as '';


DROP TABLE MRXW_FIN;
CREATE TABLE MRXW_FIN (
	LAT	char(3) NOT NULL,
	WD	varchar(100) NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXW_FIN from 'META/MRXW_FIN.RRF' with delimiter as '|' null as '';


DROP TABLE MRXW_FRE;
CREATE TABLE MRXW_FRE (
	LAT	char(3) NOT NULL,
	WD	varchar(100) NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXW_FRE from 'META/MRXW_FRE.RRF' with delimiter as '|' null as '';


DROP TABLE MRXW_GER;
CREATE TABLE MRXW_GER (
	LAT	char(3) NOT NULL,
	WD	varchar(100) NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXW_GER from 'META/MRXW_GER.RRF' with delimiter as '|' null as '';


DROP TABLE MRXW_HEB;
CREATE TABLE MRXW_HEB (
	LAT	char(3) NOT NULL,
	WD	varchar(100) NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXW_HEB from 'META/MRXW_HEB.RRF' with delimiter as '|' null as '';


DROP TABLE MRXW_HUN;
CREATE TABLE MRXW_HUN (
	LAT	char(3) NOT NULL,
	WD	varchar(100) NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXW_HUN from 'META/MRXW_HUN.RRF' with delimiter as '|' null as '';


DROP TABLE MRXW_ITA;
CREATE TABLE MRXW_ITA (
	LAT	char(3) NOT NULL,
	WD	varchar(100) NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXW_ITA from 'META/MRXW_ITA.RRF' with delimiter as '|' null as '';


DROP TABLE MRXW_JPN;
CREATE TABLE MRXW_JPN (
	LAT	char(3) NOT NULL,
	WD	varchar(100) NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXW_JPN from 'META/MRXW_JPN.RRF' with delimiter as '|' null as '';


DROP TABLE MRXW_NOR;
CREATE TABLE MRXW_NOR (
	LAT	char(3) NOT NULL,
	WD	varchar(100) NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXW_NOR from 'META/MRXW_NOR.RRF' with delimiter as '|' null as '';


DROP TABLE MRXW_POR;
CREATE TABLE MRXW_POR (
	LAT	char(3) NOT NULL,
	WD	varchar(100) NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXW_POR from 'META/MRXW_POR.RRF' with delimiter as '|' null as '';


DROP TABLE MRXW_RUS;
CREATE TABLE MRXW_RUS (
	LAT	char(3) NOT NULL,
	WD	varchar(100) NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXW_RUS from 'META/MRXW_RUS.RRF' with delimiter as '|' null as '';


DROP TABLE MRXW_SPA;
CREATE TABLE MRXW_SPA (
	LAT	char(3) NOT NULL,
	WD	varchar(100) NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXW_SPA from 'META/MRXW_SPA.RRF' with delimiter as '|' null as '';


DROP TABLE MRXW_SWE;
CREATE TABLE MRXW_SWE (
	LAT	char(3) NOT NULL,
	WD	varchar(100) NOT NULL,
	CUI	char(9) NOT NULL,
	LUI	char(9) NOT NULL,
	SUI	char(9) NOT NULL

);
\copy MRXW_SWE from 'META/MRXW_SWE.RRF' with delimiter as '|' null as '';


DROP TABLE AMBIGSUI;
CREATE TABLE AMBIGSUI (
	SUI	char(9) NOT NULL,
	CUI	char(9) NOT NULL

);
\copy AMBIGSUI from 'META/AMBIGSUI.RRF' with delimiter as '|' null as '';


DROP TABLE AMBIGLUI;
CREATE TABLE AMBIGLUI (
	LUI	char(9) NOT NULL,
	CUI	char(9) NOT NULL

);
\copy AMBIGLUI from 'META/AMBIGLUI.RRF' with delimiter as '|' null as '';
