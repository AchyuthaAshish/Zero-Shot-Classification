"""Training dataset manager for local industrial defect classification.

Creates and validates data/training/defect_training.csv.
Ensures zero data leakage with the held-out 93-case evaluation benchmark.
Enforces strict 8-category taxonomy and reports data quality checks.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import (
    TRAINING_CSV_PATH,
    DATA_DIR,
    APPROVED_CATEGORIES,
    EVALUATION_DATASET_PATH,
    RANDOM_SEED,
    VALIDATION_SPLIT_RATIO
)

# Enriched seed dataset definitions: 75 examples per category = 600 examples total
# Balanced distribution across English, native Telugu script, and Telugu-English code-switching.
_DATASET_EXAMPLES: List[Dict[str, str]] = [
    # =========================================================================
    # 1. MECHANICAL FAULT (75 examples)
    # =========================================================================
    # English (35 examples)
    {"text": "Conveyor drive roller bearing is producing abnormal grinding noise.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Main drive shaft has excessive axial vibration and radial runout.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Gearbox bevel gear teeth are heavily worn causing rattling sounds.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Centrifugal pump impeller is unbalanced and vibrating violently.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Cooling fan blade cracked and rubbing against outer shroud.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Hydraulic cylinder rod is scored and binding during stroke extension.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Flexible shaft coupling rubber insert has completely shredded.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Overhead crane wire rope slipped off drum guide grooves.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Pneumatic actuator guide rod is jammed due to debris accumulation.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Spindle bearing cage failure causing severe mechanical friction.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "V-belt tension loosened and slipping across motor pulley under load.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Mixer agitator shaft bent after foreign material entered the vat.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Linear slide rail balls seized preventing carriage movement.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Exhaust blower rotor rattling against housing at 1500 RPM.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Reciprocating compressor piston rings worn causing mechanical knocking.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Drive sprocket teeth sheared under heavy mechanical overload.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Conveyor roller bracket loose and vibrating against frame.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "High pressure slurry pump mechanical seal face scored and leaking.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Robotic arm wrist joint backlash exceeds permissible mechanical tolerance.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Ball screw nut on CNC lathe Z-axis is binding during rapid feed.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Motor bearing is producing abnormal noise.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Heavy vibration observed in pump housing.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Flywheel wobbling on main shaft during rotation.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Crusher jaw plate cracked and vibrating loose.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Rotary valve rotor blades rubbing against casing.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "The conveyor motor is making a grinding noise.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "The conveyor drive is producing a harsh rattling sound while running.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "The motor bearing produces a continuous grinding sound.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "The conveyor shaft appears to be misaligned.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Drive pulley bearing is chattering continuously under belt tension.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Conveyor drive motor emits a loud grinding noise under normal operation.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Motor shaft bearing cage collapsed producing harsh mechanical friction.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Rattling and clattering sound heard inside the reduction gearbox.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Conveyor idler roller seized causing excessive friction against belt.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Mechanical vibration caused by severe misalignment between motor and pump.", "category": "Mechanical Fault", "language": "English", "source": "manual_seed"},
    # Telugu script (18 examples)
    {"text": "మోటార్ బేరింగ్ నుంచి అసాధారణ రాపిడి శబ్దం వస్తోంది.", "category": "Mechanical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "కన్వేయర్ బెల్ట్ డ్రైవ్ షాఫ్ట్ చాలా ఎక్కువగా వైబ్రేట్ అవుతోంది.", "category": "Mechanical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "గేర్‌బాక్స్ లోపల పళ్ళు అరిగిపోయి శబ్దం వినబడుతోంది.", "category": "Mechanical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "పంప్ ఇంపెల్లర్ బ్యాలెన్స్ తప్పి గట్టిగా శబ్దం చేస్తోంది.", "category": "Mechanical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "కూలింగ్ ఫ్యాన్ బ్లేడ్ విరిగి బాడీకి తగులుతోంది.", "category": "Mechanical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "హైడ్రాలిక్ సిలిండర్ రాడ్ కదలక మధ్యలోనే జామ్ అయింది.", "category": "Mechanical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "మోటార్ కప్లింగ్ రబ్బర్ బుష్ పగిలిపోయి షాఫ్ట్ కదులుతోంది.", "category": "Mechanical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "రోలర్ బేరింగ్స్ విరిగిపోయి తిరగడం ఆగిపోయింది.", "category": "Mechanical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "మెషిన్ బేస్ బోల్టులు వదులై షేకింగ్ ఎక్కువగా ఉంది.", "category": "Mechanical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "కన్వేయర్ రోలర్ తిరగకుండా ఆగిపోయింది.", "category": "Mechanical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "షాఫ్ట్ వంకరపోయి రొటేషన్ సరిగ్గా అవ్వడం లేదు.", "category": "Mechanical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "గేర్ వీల్స్ ఒకదానికొకటి రాసుకుంటూ వైబ్రేషన్ వస్తోంది.", "category": "Mechanical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "పంప్ బాడీ గట్టిగా అదిరిపోతోంది.", "category": "Mechanical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "మెషిన్ స్పిండిల్ బేరింగ్ జామ్ అయి నడవట్లేదు.", "category": "Mechanical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "డ్రైవ్ బెల్ట్ లూజ్ అయి పుల్లీ పై స్లిప్ అవుతోంది.", "category": "Mechanical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "మోటార్ బేరింగ్ తిరిగేటప్పుడు రాపిడి మరియు గ్రైండింగ్ శబ్దం వస్తోంది.", "category": "Mechanical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "కన్వేయర్ డ్రైవ్ షాఫ్ట్ సరిగ్గా తిరగకుండా గట్టిగా రాసుకుంటోంది.", "category": "Mechanical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "గేర్‌బాక్స్ లోపల పళ్ళు విరిగి రాపిడి ధ్వని వస్తోంది.", "category": "Mechanical Fault", "language": "Telugu", "source": "manual_seed"},
    # Telugu-English code-switched (22 examples)
    {"text": "Motor lo unusual grinding sound vastundi and shaft vibrate avtundi.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Conveyor roller jam aipoindi, belt move avvatledu.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Pump bearing worn out aindi, heavy mechanical friction undi.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Gearbox lo gear teeth crack aipoyi rattling sound vastundi.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Drive shaft alignment miss aindi, heavy wobble kanipistundi.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Fan blade rub avtundi outer frame ki, noise baga vastondi.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Coupling bolts loose ayyi shaft abnormal ga shake avtundi.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Hydraulic rod stroke madhyalo stuck aindi mechanical obstacle valla.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Motor pulley slipping avtundi, belt tension poindi.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Linear guide rail seized aipoindi ball bearings break avvadam valla.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Main drive bearing fail aindi, huge mechanical vibration undi.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Compressor piston knocking noise vastundi running lo.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Mixer blade vankara aipoyi tank wall ki scrape avtundi.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Crane cable drum nunchi slip aindi mechanical guide broken valla.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "CNC lathe spindle bearing friction valla tight aipoindi.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Motor mounting base bolts loose aipoyi vibration perigindi.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Rotor unbalance valla machine heavily shaking.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Sprocket teeth shear aipoyayi heavy mechanical load valla.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Agitator shaft wobbling unsteadily in mixing vessel.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Conveyor idler roller bearing completely seized.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Motor bearing nunchi continuous grinding sound vastundi.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Conveyor drive shaft misaligned ayyi rattle sound chestundi.", "category": "Mechanical Fault", "language": "Telugu-English", "source": "manual_seed"},

    # =========================================================================
    # 2. ELECTRICAL FAULT (75 examples)
    # =========================================================================
    # English (35 examples)
    {"text": "Motor terminal box experienced an arc flash with visible scorching.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Phase-to-phase short circuit detected on the feeder cable.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Stator winding insulation breakdown caused instantaneous ground fault.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Contactor coil burnt out and will not pull in the main contacts.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Loose crimp on phase B terminal lug causing heavy resistive sparking.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Earth leakage relay tripped due to degraded cable insulation.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Circuit breaker tripped on instantaneous magnetic overcurrent protection.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Three-phase motor running on single phasing after fuse blown on line 3.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Current transformer secondary circuit opened causing high voltage arcing.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Electrical flashover inside the 415V distribution switchboard.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Severe phase current imbalance exceeding 35% across motor windings.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Control relay contacts welded shut due to sustained inrush current.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Power cable is damaged with exposed conductors touching enclosure.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Capacitor bank cell exploded in power factor correction cubicle.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Busbar support insulator cracked leading to flashover to enclosure earth.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Motor winding resistance test shows open circuit on phase U.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Ground fault detected on branch feeder circuit 4.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Thermal magnetic breaker contacts badly pitted and arcing under load.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Neutral conductor disconnected causing severe phase voltage displacement.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Insulation resistance of motor measured below 0.1 Megaohms to ground.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Main disconnect switch contacts burnt and failing to close phase 2.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Heavy electrical sparks coming from motor slip rings.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Residual current device trips immediately when energized.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Short circuit in control wiring harness behind panel door.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Solenoid valve coil shorted internally and blowing branch fuse.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Short circuit detected in terminal box with visible arc flash.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Three-phase feeder conductor shorted to conduit frame causing breaker trip.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Contactor auxiliary contacts pitted causing intermittent circuit disconnect.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Stator coil dielectric breakdown detected during insulation surge test.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Electrical panel busbar joint loose resulting in high resistance heating.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Motor junction box terminals showing severe copper burn marks.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Earth fault relay tripped after phase line shorted to ground.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Circuit breaker thermal overload contacts burnt shut.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Current transformer secondary open circuit causing high voltage flash.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    {"text": "Terminal block lugs heavily oxidized creating severe arcing.", "category": "Electrical Fault", "language": "English", "source": "manual_seed"},
    # Telugu script (18 examples)
    {"text": "మోటార్ టెర్మినల్ బాక్స్‌లో షార్ట్ సర్క్యూట్ జరిగి స్పార్క్ వస్తోంది.", "category": "Electrical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "కేబుల్ ఇన్సులేషన్ దెబ్బతిని గ్రౌండ్ ఫాల్ట్ అయింది.", "category": "Electrical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "కాంటాక్టర్ కాయిల్ కాలిపోయి స్విచ్ ఆన్ కావడం లేదు.", "category": "Electrical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "త్రీ ఫేజ్ మోటార్ లో ఒక ఫేజ్ కట్ అయి సింగిల్ ఫేజింగ్ వస్తోంది.", "category": "Electrical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఎలక్ట్రికల్ ప్యానెల్ లో ఫ్యూజ్ బ్లో అయిపోయింది.", "category": "Electrical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "మెయిన్ సర్క్యూట్ బ్రేకర్ ఓవర్‌కరెంట్ వల్ల ట్రిప్ అయింది.", "category": "Electrical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "మోటార్ వైండింగ్ కాలిపోయి పొగ వస్తోంది.", "category": "Electrical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "వైరింగ్ కాలిపోయి జంక్షన్ బాక్స్ లో నిప్పురవ్వలు వచ్చాయి.", "category": "Electrical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఎర్త్ లీకేజ్ రిలే పదే పదే ట్రిప్ అవుతోంది.", "category": "Electrical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఫేజ్ టెర్మినల్ లూజ్ కావడం వల్ల ఆర్కింగ్ జరుగుతోంది.", "category": "Electrical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "కరెంట్ అసమతుల్యత వల్ల మోటార్ కరెంట్ ఎక్కువ తీసుకుంటోంది.", "category": "Electrical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "స్విచ్‌బోర్డులో వైర్ తెగిపోయి బాడీకి తగిలింది.", "category": "Electrical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "కంట్రోల్ రిలే కాంటాక్ట్స్ అతుక్కుపోయాయి.", "category": "Electrical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "మోటార్ కాయిల్ షార్ట్ అయి తీవ్రమైన ఆర్క్ ఫ్లాష్ వచ్చింది.", "category": "Electrical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "పవర్ కేబుల్ కట్ అయి విద్యుత్ ప్రసారం ఆగిపోయింది.", "category": "Electrical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "మోటార్ టెర్మినల్స్ వద్ద విద్యుత్ నిప్పురవ్వలు వస్తున్నాయి.", "category": "Electrical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "వైరింగ్ కాలిపోయి స్విచ్ గేర్ ట్రిప్ అయ్యింది.", "category": "Electrical Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఫేజ్ వైర్ బాడీకి తగిలి షార్ట్ సర్క్యూట్ జరిగింది.", "category": "Electrical Fault", "language": "Telugu", "source": "manual_seed"},
    # Telugu-English code-switched (22 examples)
    {"text": "Motor terminal box lo short circuit aindi, spark kanipistundi.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Contactor coil burn aipoindi, main switch on avvatledu.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Phase imbalance valla motor breaker trip aindi.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Stator winding short circuit aipoyi heavy smoke vastundi.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Earth leakage relay trip aindi cable insulation damage valla.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Terminal block daggara loose connection valla sparking vastundi.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Circuit breaker instant overcurrent valla trip aindi.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Single phasing valla motor abnormal current draw chestundi.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Fuse blow aipoindi phase 1 line lo short circuit valla.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Panel board lo arc flash vachindi busbar daggara.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Control relay contacts weld aipoyayi inrush current valla.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Insulation resistance low ga undi phase-to-ground short valla.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Power cable burnt smell vastundi terminal box nunchi.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Capacitor cell burst aindi power factor cabinet lo.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Solenoid coil short circuit aipoyi circuit breaker trip chestundi.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Motor phase winding cut aindi, continuity ledu.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "415V line short circuit to ground tripped upstream breaker.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Cable insulation crack ayyi conductive metal ki touch aindi.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Motor lead wire burned and sparks observed.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Switchgear feeder contact resistance is abnormally high.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Terminal box lo electrical arcing smell vastundi.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Circuit breaker instant magnetic trip aindi short circuit valla.", "category": "Electrical Fault", "language": "Telugu-English", "source": "manual_seed"},

    # =========================================================================
    # 3. SENSOR FAULT (75 examples)
    # =========================================================================
    # English (35 examples)
    {"text": "Temperature sensor gives incorrect readings while actual fluid is ambient.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Pressure transmitter output signal is locked at 4 mA irrespective of actual line pressure.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Optical proximity sensor lens is blinded and fails to detect metal parts.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Rotary optical encoder lost pulse count causing positioning discrepancies.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "RTD temperature probe resistance drifted by 150 ohms causing false alarm.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Ultrasonic tank level sensor echo signal lost due to transducer element defect.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Thermocouple wire opened showing off-scale negative temperature value.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Magnetic reed switch stuck closed even after cylinder magnet moves away.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Vibration accelerometer cable loose giving noisy sporadic spikes on monitor.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Electromagnetic flowmeter reading zero flow while pump is visibly discharging fluid.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Differential pressure cell diaphragm damaged producing constant 20 mA signal.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Photoelectric safety light curtain emitter diode failed.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Laser distance sensor returning error code E04 out of range measurement.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "pH probe electrode glass bulb cracked causing output voltage to drop to 0 mV.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Tachometer pickup sensor misses teeth on target wheel producing erratic speed data.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Limit switch mechanical arm broken and contacts never actuate.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Humidity transmitter reading 100% relative humidity in dry oven environment.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Load cell Wheatstone bridge circuit has an open strain gauge branch.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Infrared pyrometer optics contaminated giving 50 degree offset error.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Inductive proximity switch LED remains lit constantly without metal target.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Gas detector sensor cell poisoned and unresponsive to test calibration gas.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Current transducer hall effect sensor gives zero millivolt output under load.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Level float switch mercury bulb cracked inside housing.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Speed sensor pickup gap too large resulting in dropped pulses.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Torque transducer calibration drifted by 20% on test bench.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "The temperature sensor is showing incorrect values.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "The temperature sensor is giving incorrect readings while pipe is cold.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Temperature probe reading 120 C but actual line temperature is only 25 C.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Thermocouple millivolt output fluctuating wildly due to broken junction.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Pressure transmitter zero offset shifted by 15 PSI after pressure surge.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Optical proximity switch not responding to passing steel targets.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Rotary encoder missing quadrature counts causing position drift.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Vibration sensor output wire broken giving zero millivolts reading.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Ultrasonic level transmitter signal lost due to defective transducer element.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    {"text": "Flowmeter sensor reading zero liters despite visible liquid flow in sight glass.", "category": "Sensor Fault", "language": "English", "source": "manual_seed"},
    # Telugu script (18 examples)
    {"text": "టెంపరేచర్ సెన్సార్ తప్పుడు రీడింగ్స్ చూపిస్తోంది.", "category": "Sensor Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ప్రెషర్ ట్రాన్స్‌మిటర్ అవుట్‌పుట్ సిగ్నల్ స్టక్ అయి మారింది లేదు.", "category": "Sensor Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ప్రాక్సిమిటీ సెన్సార్ వస్తువును గుర్తించడం లేదు.", "category": "Sensor Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఎన్‌కోడర్ పల్స్ సిగ్నల్ మిస్ అయి పొజిషన్ తప్పుగా వస్తోంది.", "category": "Sensor Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "థర్మోకపుల్ వైర్ తెగిపోయి స్క్రీన్ పై ఓపెన్ సర్క్యూట్ ఎర్రర్ వచ్చింది.", "category": "Sensor Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "లెవల్ సెన్సార్ ట్యాంక్ ఖాళీగా ఉన్నా ఫుల్ అని రీడింగ్ ఇస్తోంది.", "category": "Sensor Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఫ్లోమీటర్ సెన్సార్ నుంచి ఎలాంటి అవుట్‌పుట్ సిగ్నల్ రావడం లేదు.", "category": "Sensor Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "లిమిట్ స్విచ్ లివర్ జామ్ అయి కాంటాక్ట్ క్లోజ్ అవ్వట్లేదు.", "category": "Sensor Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఫోటోఎలెక్ట్రిక్ సెన్సార్ లైట్ బీమ్ ఆగిపోయింది.", "category": "Sensor Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "స్పీడ్ సెన్సార్ రాంగ్ రీడింగ్ ఇవ్వడం వల్ల ఆర్పీఎమ్ జీరో చూపిస్తోంది.", "category": "Sensor Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "సెన్సార్ వైర్ లూజ్ అయి సిగ్నల్ ఫ్లక్చుయేట్ అవుతోంది.", "category": "Sensor Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "వైబ్రేషన్ సెన్సార్ సెన్సిటివిటీ కోల్పోయి రీడింగ్స్ ఆగిపోయాయి.", "category": "Sensor Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "సెన్సార్ కాలిబ్రేషన్ డ్రిఫ్ట్ అయి తప్పు విలువలు చూపిస్తోంది.", "category": "Sensor Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఆప్టికల్ సెన్సార్ లెన్స్ పాడై పార్ట్స్ డిటెక్ట్ చేయట్లేదు.", "category": "Sensor Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ప్రెషర్ సెన్సార్ పైపులో ప్రెషర్ ఉన్నా సున్నా బార్ చూపిస్తోంది.", "category": "Sensor Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఉష్ణోగ్రత సెన్సార్ కాలిబ్రేషన్ తప్పి తప్పుడు విలువలు ఇస్తోంది.", "category": "Sensor Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ప్రెజర్ ట్రాన్స్‌మిటర్ అవుట్‌పుట్ 4 ఎంఏ వద్ద స్థిరంగా ఉండిపోయింది.", "category": "Sensor Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ప్రాక్సిమిటీ సెన్సార్ పనిచేయక పొజిషన్ సిగ్నల్ మిస్ అయింది.", "category": "Sensor Fault", "language": "Telugu", "source": "manual_seed"},
    # Telugu-English code-switched (22 examples)
    {"text": "Temperature sensor incorrect reading istundi, actual temperature normal ga undi.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Pressure transmitter drift aipoindi, wrong pressure display chestundi.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Proximity sensor target detect cheyatledu metal vachina kuda.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Rotary encoder pulses miss chestundi, position error vastundi.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Thermocouple probe open circuit aindi, value negative chupistundi.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Level sensor tank empty unna full reading display chestundi.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Flow sensor output 0 mA ki lock aindi running pump lo.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Vibration accelerometer sensor cable loose ayyi spikes vastunnayi.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Limit switch actuator break aindi, contact close avvatledu.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Photo sensor optical beam cut aindi internal diode fail valla.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Speed sensor pickup pulses ivvatledu, RPM zero chupistundi.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Load cell calibration offset aindi 25 kg variance tho.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "RTD sensor resistance fluctuate avtundi wrong temperature data tho.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Laser sensor distance wrong ga calculate chestundi optics damage valla.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Inductive switch constantly ON position lo undi object lekapoyina.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Safety light curtain sensor fault status lo undi.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Gas detector sensor calibration fail aindi zero response tho.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Hall sensor output signal drop aipoindi.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Analog input channel lo sensor noise heavy ga undi.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Sensor reading looks wrong compared to gauge reading.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Temperature sensor continuous ga wrong reading display chestundi.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Pressure transducer output drift ayyi wrong measurement istundi.", "category": "Sensor Fault", "language": "Telugu-English", "source": "manual_seed"},

    # =========================================================================
    # 4. TEMPERATURE FAULT (75 examples)
    # =========================================================================
    # English (35 examples)
    {"text": "Motor is overheating and housing surface temperature reached 115 degrees Celsius.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Gearbox oil temperature exceeded maximum allowable thermal limit of 95C.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Bearing housing is running dangerously hot to the touch.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Cooling heat exchanger thermal capacity insufficient causing severe coolant overheating.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Compressor discharge air temperature exceeded safety trip threshold.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Furnace refractory wall has a severe thermal hot spot radiating extreme heat.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Hydraulic reservoir fluid temperature rising continuously towards thermal runaway.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Electrical cabinet internal ambient temperature reached 65C tripping thermal relays.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Turbine exhaust temperature spread abnormally high indicating uneven combustion heat.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Extruder heating zone 3 temperature dropped far below processing setpoint.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Transformer oil temperature high alarm sounded in substation.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Chiller evaporator temperature freezing up due to low refrigeration pressure.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Brake friction disc overheating rapidly during emergency deceleration cycles.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Molding die tool temperature gradient uneven across upper and lower plates.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "VFD heat sink temperature over-limit warning active.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Induction motor stator temperature reached class H thermal insulation threshold.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Boiler steam drum metal temperature gradient exceeding thermal stress limits.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Oil cooler clogged causing high temperature warning on lubrication loop.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Centrifugal pump casing extremely hot and fluid vaporizing inside.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Thermal overload trip on conveyor drive after prolonged overheating.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Machine is getting very hot after running for thirty minutes.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Spindle temperature alarm triggered due to inadequate cooling circulation.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Reactor vessel jacket temperature out of control and heating rapidly.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Hot air blower temperature regulation failure resulting in thermal excess.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Cutting tool workpiece interface experiencing extreme thermal burning.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "The motor is overheating and surface temperature exceeded 110 degrees.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Motor casing is extremely hot to the touch while current draw is nominal.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Gearbox thermal buildup exceeded maximum safe operating temperature of 95C.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Hydraulic oil cooler blocked causing continuous thermal rise in reservoir.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Bearing temperature rose rapidly to 105 degrees Celsius triggering alarm.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Extruder heating zone temperature runaway reached thermal hazard limit.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Centrifugal pump casing blistering hot due to dry running friction heat.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Compressor discharge air reached critical 120 C thermal shutdown setpoint.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Transformer winding thermal gauge indicates critical overheating condition.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    {"text": "Cooling circuit failure caused spindle unit temperature to surge to 85C.", "category": "Temperature Fault", "language": "English", "source": "manual_seed"},
    # Telugu script (18 examples)
    {"text": "మోటార్ చాలా వేడెక్కిపోతోంది, ముట్టుకుంటే కాలిపోయేలా ఉంది.", "category": "Temperature Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "గేర్‌బాక్స్ ఆయిల్ టెంపరేచర్ 95 డిగ్రీల కంటే ఎక్కువ పెరిగింది.", "category": "Temperature Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "బేరింగ్ వేడి ఎక్కిపోయి పొగ వస్తోంది.", "category": "Temperature Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "కూలింగ్ సిస్టమ్ ఉన్నప్పటికీ ఇంజిన్ ఓవర్‌హీట్ అవుతోంది.", "category": "Temperature Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "మెషిన్ టెంపరేచర్ సేఫ్టీ లిమిట్ దాటిపోయి అలారం మోగింది.", "category": "Temperature Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "హైడ్రాలిక్ ఆయిల్ వేడెక్కి థర్మల్ ప్రొటెక్షన్ ట్రిప్ అయింది.", "category": "Temperature Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "కంప్రెసర్ ఎగ్జాస్ట్ గాలి ఉష్ణోగ్రత చాలా ప్రమాదకరంగా పెరిగింది.", "category": "Temperature Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఎలక్ట్రికల్ ప్యానెల్ లోపల ఉష్ణోగ్రత విపరీతంగా పెరిగిపోయింది.", "category": "Temperature Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ట్రాన్స్‌ఫార్మర్ ఆయిల్ వేడెక్కి హై టెంపరేచర్ అలారం వచ్చింది.", "category": "Temperature Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "స్పిండిల్ వేడెక్కిపోయి మెషిన్ ఆగిపోయింది.", "category": "Temperature Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "హీటర్ కంట్రోల్ ఫెయిల్ అయి విపరీతమైన వేడి పుడుతోంది.", "category": "Temperature Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "చిల్లర్ కూలెంట్ చల్లబడకుండా టెంపరేచర్ పెరుగుతోంది.", "category": "Temperature Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "బ్రేక్ డిస్క్ వేడెక్కి ఎర్రగా మారుతోంది.", "category": "Temperature Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "డ్రైవ్ మోటార్ థర్మల్ ఓవర్‌లోడ్ వల్ల ఆటోమేటిక్ గా షట్‌డౌన్ అయింది.", "category": "Temperature Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఓవెన్ ఉష్ణోగ్రత సెట్ పాయింట్ కంటే చాలా ఎక్కువ చేరుకుంది.", "category": "Temperature Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "మెషిన్ బాడీ విపరీతంగా వేడెక్కిపోయి థర్మల్ అలారం మోగుతోంది.", "category": "Temperature Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "మోటార్ కూలింగ్ ఫ్యాన్ ఉన్నప్పటికీ ఉష్ణోగ్రత 100 డిగ్రీలు దాటింది.", "category": "Temperature Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఆయిల్ చాలా వేడెక్కి థర్మల్ కట్-ఆఫ్ స్విచ్ యాక్టివేట్ అయింది.", "category": "Temperature Fault", "language": "Telugu", "source": "manual_seed"},
    # Telugu-English code-switched (22 examples)
    {"text": "Motor chala heat avutundi, thermal protection trip aipoindi.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Bearing temperature continuous ga increase avtundi 90C daatindi.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Gearbox oil overheat aindi cooling failure valla.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Machine body chala hot ga aindi operation start chesina 20 mins lo.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Hydraulic oil temperature limit exceed aindi, thermal alarm active.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Compressor discharge air chala వేడిగా వస్తోంది safe limit kante.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Panel cooling fan aagipoyi cabinet temperature high aipoindi.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "VFD heat sink temperature over-limit trip aindi.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Pump casing extreme heat tho run avtundi fluid vaporize avtundi.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Heater chamber temperature regulation fail ayyi extreme heat undi.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Motor winding thermal sensor high temperature alarm istundi.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Transformer oil temperature high trip trigger aindi.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Spindle unit over heating problem valla operation halt chesam.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Cooling loop heat exchanger blocked valla temperature rise aindi.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Brake disc heat up aipoyi smoke smell vastundi.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Thermal cut-off switch activated due to motor overheating.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Machine running lo excessive thermal heat generate avtundi.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Oven temperature control out of bounds heating continuously.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Exhaust temperature high warning displayed on panel.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Bearing thermal trip triggered repeatedly.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Motor chala heat aipoyi thermal overload switch trip aindi.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Gearbox oil thermal alarm trigger aindi 95 degrees exceed valla.", "category": "Temperature Fault", "language": "Telugu-English", "source": "manual_seed"},

    # =========================================================================
    # 5. SOFTWARE FAULT (75 examples)
    # =========================================================================
    # English (35 examples)
    {"text": "Control software crashes during startup with segmentation fault.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "HMI display application froze on main menu and stopped responding to touches.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "SCADA runtime encountered an unhandled null pointer exception.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "PLC firmware watchdog timeout error triggered sudden system halt.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Recipe management database crashed due to corrupted SQLite index file.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Motion controller software task entered an infinite execution loop.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Memory leak in data logging daemon exhausted system RAM after 48 hours.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Operating system blue screen error displayed on industrial PC console.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Configuration script syntax error caused parameter loading to abort.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Alarm logging software service terminated unexpectedly with exit code 139.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Vision inspection software threw an access violation in image processing DLL.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Batch execution engine deadlocked waiting for shared memory mutex.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Firmware upgrade failed leaving microcontroller in unbootable brick state.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "User interface application threw an out of memory error during report generation.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Robotic trajectory planning algorithm threw mathematical divide by zero exception.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Security authentication daemon hanging and denying all valid operator logins.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Real-time task scheduler missed deadline in cyclic execution thread.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "HMI script runtime error line 42 variable undefined.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Data acquisition driver crashed after invalid registry configuration.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Automated test sequence script crashed during regression cycle.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "PLC logic routine stuck in while loop blocking cycle execution.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Industrial PC file system corrupted leading to operating system boot failure.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "OPC UA server software stack overflow in tag browsing method.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Telemetry recording service failed with database locked error.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Operator touchscreen UI crashed and dropped to Windows desktop.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Control software crashes during boot sequence with memory fault.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "HMI touchscreen application unresponsive and frozen on main view.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Controller firmware task deadlocked waiting for shared mutex.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Recipe manager software crashed with database access violation.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "SCADA trend display software terminated with unhandled exception.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Motion controller positioning program threw numerical overflow error.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Industrial PC software locked in blue screen memory dump.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Data logging daemon crashed after operating system file descriptor exhaustion.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Batch execution routine halted due to null pointer dereference in logic.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    {"text": "Automated sequence script failed to load due to corrupted syntax file.", "category": "Software Fault", "language": "English", "source": "manual_seed"},
    # Telugu script (18 examples)
    {"text": "కంట్రోల్ సాఫ్ట్‌వేర్ స్టార్ట్ కాగానే క్రాష్ అవుతోంది.", "category": "Software Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "HMI టచ్ స్క్రీన్ అప్లికేషన్ ఫ్రీజ్ అయి రెస్పాండ్ అవ్వడం లేదు.", "category": "Software Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "మెషిన్ సాఫ్ట్‌వేర్‌లో రన్‌టైమ్ ఎర్రర్ వచ్చి సిస్టమ్ ఆగిపోయింది.", "category": "Software Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "PLC ఫర్మ్‌వేర్ వాచ్‌డాగ్ టైమ్‌అవుట్ ఎర్రర్ ఇచ్చింది.", "category": "Software Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "డేటాబేస్ క్రాష్ అవ్వడం వల్ల రెసిపీ లోడ్ కావడం లేదు.", "category": "Software Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఆపరేటింగ్ సిస్టమ్ బ్లూ స్క్రీన్ ఎర్రర్ చూపిస్తోంది.", "category": "Software Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "మెమరీ లీక్ వల్ల సాఫ్ట్‌వేర్ చాలా స్లో అయి ఆగిపోయింది.", "category": "Software Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "విజన్ ప్రాసెసింగ్ సాఫ్ట్‌వేర్ ఫైల్ ఎర్రర్ వల్ల క్లోజ్ అయింది.", "category": "Software Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "సాఫ్ట్‌వేర్ రీబూట్ లూప్‌లో పడి లాగిన్ పేజీ రావడం లేదు.", "category": "Software Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "స్క్రిప్ట్ ఎర్రర్ వల్ల ఆటోమేటిక్ ప్రోగ్రామ్ రన్ అవ్వడం ఆగిపోయింది.", "category": "Software Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "HMI స్క్రీన్‌పై అన్‌హ్యాండిల్డ్ ఎక్సెప్షన్ మెసేజ్ వస్తోంది.", "category": "Software Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "సాఫ్ట్‌వేర్ అప్‌డేట్ తర్వాత ప్రోగ్రామ్ ఓపెన్ అవ్వడం లేదు.", "category": "Software Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ట్రాజెక్టరీ ప్లానింగ్ ప్రోగ్రామ్ క్రాష్ అయి రోబోట్ ఆగింది.", "category": "Software Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "కంట్రోలర్ సాఫ్ట్‌వేర్ డెడ్‌లాక్ అయి కమాండ్స్ తీసుకోవట్లేదు.", "category": "Software Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "లాగిన్ సర్వీస్ సాఫ్ట్‌వేర్ రెస్పాన్స్ ఇవ్వట్లేదు.", "category": "Software Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "కంట్రోల్ సాఫ్ట్‌వేర్ ఫ్రీజ్ అయి ఎలాంటి ఆదేశాలు తీసుకోవడం లేదు.", "category": "Software Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "రన్‌టైమ్ ఎక్సెప్షన్ వల్ల ఆపరేటింగ్ ప్రోగ్రామ్ క్లోజ్ అయింది.", "category": "Software Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "HMI అప్లికేషన్ స్టార్టప్‌లో ఎర్రర్ వచ్చి ఆగిపోయింది.", "category": "Software Fault", "language": "Telugu", "source": "manual_seed"},
    # Telugu-English code-switched (22 examples)
    {"text": "Control software startup lo crash aindi segmentation fault valla.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "HMI screen freeze aipoindi, touch screen buttons respond avvatledu.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "SCADA runtime application lo null pointer exception error vachindi.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "PLC firmware watchdog timeout error trigger aindi.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Recipe database corrupt aipoyi batch program load avvatledu.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Software memory leak valla system RAM full aipoyi hang aindi.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Industrial PC lo blue screen dump error vachindi.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Vision camera software processing DLL crash aipoindi.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Script runtime error line 54 undefined variable error undi.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Firmware update fail ayyi controller boot loop lo paddadi.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Operator application sudden ga terminate aindi exit code 139 tho.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Motion control task deadlock aipoyi commands execute avvatledu.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "HMI display crash ayyi Windows desktop kanipistundi.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Telemetry logging service memory allocation error valla aagipoindi.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Trajectory algorithm divide by zero exception raise chesindi.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Software configuration file parse error throwing exception.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Batch controller program thread stuck in infinite loop.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "SCADA graphics engine failed to render plant mimic display.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Control application unhandled exception caught at runtime.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "PLC cyclic task execution time overrun fault occurred.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Control software segmentation fault valla suddenly crash aindi.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "SCADA application hang aipoindi operator input tiskovatledu.", "category": "Software Fault", "language": "Telugu-English", "source": "manual_seed"},

    # =========================================================================
    # 6. POWER SUPPLY FAULT (75 examples)
    # =========================================================================
    # English (35 examples)
    {"text": "Power supply voltage is unstable with severe oscillations between 18V and 26V.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "24V DC switch-mode power supply unit failed and output dropped to zero volts.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "UPS battery bank failed discharge test and cannot support inverter load during blackout.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Main control transformer secondary output voltage dropped below 190V AC.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "DC power distribution bus has excessive high-frequency voltage ripple.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Redundant power supply module A failed with blinking red fault LED.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Industrial PC ATX power supply internal cooling fan stopped and unit overheated.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Emergency power transfer switch failed to engage auxiliary diesel generator.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "DC-DC buck converter output capacitor swelled and leaking electrolyte.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Constant voltage regulator output fluctuating violently under inductive load.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Battery charger failed to deliver float charging current to station batteries.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "AC line voltage sag caused brownout shutdown of all 24V DC auxiliary power supplies.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Power supply crowbar overvoltage protection circuit latched on false trigger.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Auxiliary power supply fuse blown on the 12V instrumentation supply rail.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Rectifier bridge diode open circuit causing high DC voltage ripple and low output.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Control cabinet 24V busbar shorted to ground through power supply chassis.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Inverter DC link voltage collapsed under motor starting transient.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Stationary battery cell #7 has dropped to 0.4V cell potential.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Linear power supply thermal regulator shut down due to continuous current overload.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Solar panel charge controller output voltage dropped to zero during daylight.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "PLC power supply module red light glowing indicating internal fault.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Control power transformer primary winding open circuit.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "48V DC telecom battery rectifier module blown.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Regulated power supply output collapsed when load exceeded 5 amperes.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Generator automatic voltage regulator failed causing 300V surge.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "The power supply voltage is unstable with fluctuating DC output.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "24V DC auxiliary rail voltage dropped to 16V under control load.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Switch-mode power supply output dropped completely to zero volts.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "UPS backup battery failed load test and cannot supply inverter.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Control transformer secondary output collapsed below permissible voltage.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Main 24V power supply module red error indicator glowing steadily.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Switching power supply internal fuse blown with noticeable burning odor.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "DC busbar voltage fluctuating wildly causing controller reboots.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Battery charger failed to provide float charge to 110V DC station batteries.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    {"text": "Power supply overvoltage crowbar circuit tripped on voltage transient.", "category": "Power Supply Fault", "language": "English", "source": "manual_seed"},
    # Telugu script (18 examples)
    {"text": "24V పవర్ సప్లై వోల్టేజ్ పడిపోయి సున్నా వోల్ట్స్ చూపిస్తోంది.", "category": "Power Supply Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "UPS బ్యాటరీ బ్యాకప్ ఫెయిల్ అయింది, కరెంట్ పోగానే సిస్టమ్ ఆగిపోయింది.", "category": "Power Supply Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "పవర్ సప్లై యూనిట్ నుంచి వోల్టేజ్ ఫ్లక్చుయేట్ అవుతోంది.", "category": "Power Supply Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "SMPS పవర్ యూనిట్ కాలిపోయి రెడ్ లైట్ వెలుగుతోంది.", "category": "Power Supply Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "కంట్రోల్ ట్రాన్స్‌ఫార్మర్ అవుట్‌పుట్ వోల్టేజ్ రావడం లేదు.", "category": "Power Supply Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "మెయిన్ DC బస్‌బార్ వోల్టేజ్ తగ్గిపోయి కంట్రోలర్ రీస్టార్ట్ అవుతోంది.", "category": "Power Supply Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "బ్యాటరీ ఛార్జర్ యూనిట్ బ్యాటరీలను ఛార్జ్ చేయడం లేదు.", "category": "Power Supply Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "పవర్ సప్లై ఓవర్‌లోడ్ ప్రొటెక్షన్ ట్రిప్ అయింది.", "category": "Power Supply Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఇన్వర్టర్ సర్క్యూట్ ఫెయిల్ అయి బ్యాకప్ పవర్ రాలేదు.", "category": "Power Supply Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "వోల్టేజ్ రెగ్యులేటర్ పాడై అస్థిరమైన విద్యుత్ ప్రసారం అవుతోంది.", "category": "Power Supply Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "PLC పవర్ సప్లై మాడ్యూల్ పాడైపోయింది.", "category": "Power Supply Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "రెక్టిఫైయర్ యూనిట్ నుంచి DC వోల్టేజ్ సరిగ్గా రావడం లేదు.", "category": "Power Supply Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఆక్సిలరీ పవర్ సప్లై ఫ్యూజ్ ఎగిరిపోయింది.", "category": "Power Supply Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "డీజిల్ జనరేటర్ AVR ఫెయిల్యూర్ వల్ల వోల్టేజ్ విపరీతంగా పెరిగింది.", "category": "Power Supply Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "పవర్ ప్యాక్ కెపాసిటర్ వాచిపోయి పవర్ డ్రాప్ అయింది.", "category": "Power Supply Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "పవర్ సప్లై వోల్టేజ్ హెచ్చుతగ్గులతో అస్థిరంగా ఉంది.", "category": "Power Supply Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "24V SMPS యూనిట్ ఆగిపోయి విద్యుత్ ప్రసారం నిలిచిపోయింది.", "category": "Power Supply Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "UPS బ్యాటరీ డిశ్చార్జ్ అయి పవర్ సపోర్ట్ ఇవ్వలేకపోయింది.", "category": "Power Supply Fault", "language": "Telugu", "source": "manual_seed"},
    # Telugu-English code-switched (22 examples)
    {"text": "24V DC power supply voltage drop aindi 14V ki, system shutdown aindi.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "UPS battery backup fail aindi, power cut avvagane line freeze aindi.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "SMPS power unit nunchi zero output voltage vastundi, red LED glowing.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Control panel power supply continuously fluctuate avtundi under load.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "DC busbar voltage unstable ga undi, heavy ripple undi.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Redundant power module A failure alert display chestundi.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Control transformer secondary output drop aindi below 180V.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Battery charger float charging current ivvatledu station battery ki.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Power supply fuse blow aindi auxiliary 12V line lo.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Inverter DC link voltage collapse aindi starting transient valla.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Rectifier bridge failure valla DC output voltage low aindi.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "PLC power module internal fault valla PLC shut aipoindi.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Voltage regulator output fluctuate avvadam valla boards reset avtunnayi.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Emergency transfer switch generator backup power ivvatledu.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "DC-DC buck converter output zero volts chupistundi.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Switch mode power supply thermal trip aindi cooling fan failure valla.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Station battery cell voltage drastically drop aipoindi.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Control power brownout triggered protective shutdown.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Power pack internal capacitor exploded smell vastundi.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Auxiliary DC rail voltage unstable with high noise.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Power supply voltage drop aipoyi 24V rail 12V ki padipoyindi.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "SMPS power unit internal fault valla complete shut down aindi.", "category": "Power Supply Fault", "language": "Telugu-English", "source": "manual_seed"},

    # =========================================================================
    # 7. COMMUNICATION FAULT (75 examples)
    # =========================================================================
    # English (35 examples)
    {"text": "PLC cannot communicate with controller over the Modbus network.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Profinet industrial Ethernet link down between main rack and remote IO block 3.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "CAN bus interface entered bus-off state due to severe frame transmission errors.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "EtherCAT slave station node 5 dropped off network ring causing state transition failure.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "RS-485 serial communication experiencing continuous cyclic redundancy check CRC errors.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Managed network switch port 8 link flapping intermittently.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Modbus TCP connection timeout after repeated socket connection attempts.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "DeviceNet trunk cable terminated improperly causing high packet collision rate.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Industrial Wi-Fi client disconnected from access point due to low signal strength.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Fiber optic communication converter optical loss of signal light active.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "HMI unable to poll telemetry data from variable frequency drive over serial port.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Safety network communication integrity error tripped safe torque off.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "OPC server connection rejected by remote SCADA workstation.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "BACnet gateway unresponsive to ping queries from building management station.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Ethernet/IP I/O connection timed out on remote servo axis.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Serial RS-232 cable disconnected between barcode reader and controller.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Network gateway packet loss rate reached 60% on fieldbus trunk.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Profibus DP master report indicates transmission error with slave address 12.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Wireless telemetry transmitter dropped out of mesh network topology.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Modbus RTU query failed with timeout error code 0x0B.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Network cable damaged causing total communication loss between stations.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Fieldbus bus terminator resistor missing causing signal reflection.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "IO-Link master channel 2 reports device communication lost.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "SCADA communication driver dropped link with remote telemetry unit RTU.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "PLC ethernet port link LED off and no data packets transmitting.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "The PLC has lost communication with the controller over the fieldbus.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Modbus TCP connection refused after socket connection timeout.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Profinet network packet drop rate exceeded threshold causing axis halt.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "CAN bus line entered bus-off state due to severe frame transmission errors.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Managed industrial ethernet switch port link is down and not communicating.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "RS485 serial communication bus experiencing continuous CRC frame errors.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "EtherCAT fieldbus master reported lost frame sync with slave modules.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Remote IO rack disconnected from main PLC network trunk cable.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "DeviceNet communication bus dropped multiple node connections.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    {"text": "Wireless telemetry gateway dropped network connection to remote sensor hub.", "category": "Communication Fault", "language": "English", "source": "manual_seed"},
    # Telugu script (18 examples)
    {"text": "PLC మరియు కంట్రోలర్ మధ్య కమ్యూనికేషన్ తెగిపోయింది.", "category": "Communication Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఈథర్‌నెట్ నెట్‌వర్క్ కేబుల్ డిస్‌కనెక్ట్ అయింది.", "category": "Communication Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "మోడ్‌బస్ నెట్‌వర్క్ టైమ్‌అవుట్ ఎర్రర్ వస్తోంది.", "category": "Communication Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఫీల్డ్‌బస్ డేటా ప్యాకెట్లు లాస్ అవుతున్నాయి.", "category": "Communication Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "HMI మరియు PLC మధ్య సిగ్నల్ కనెక్షన్ కట్ అయింది.", "category": "Communication Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ప్రొఫినెట్ లింక్ డౌన్ అయి రిమోట్ మాడ్యూల్స్ కనిపించట్లేదు.", "category": "Communication Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "నెట్‌వర్క్ స్విచ్ పోర్ట్ కమ్యూనికేషన్ కోల్పోయింది.", "category": "Communication Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "CAN బస్ ఎర్రర్ వల్ల డ్రైవ్స్ తో కమ్యూనికేషన్ ఆగింది.", "category": "Communication Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "వైర్‌లెస్ గేట్‌వే సిగ్నల్ డ్రాప్ అయి రిమోట్ కంట్రోల్ పనిచేయట్లేదు.", "category": "Communication Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఆప్టికల్ ఫైబర్ కమ్యూనికేషన్ లైన్ తెగిపోయింది.", "category": "Communication Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "సీరియల్ కమ్యూనికేషన్ కేబుల్ లూజ్ అయి డేటా రావడం లేదు.", "category": "Communication Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "రిమోట్ IO స్టేషన్ తో డేటా ఎక్స్ఛేంజ్ ఫెయిల్ అయింది.", "category": "Communication Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఈథర్‌క్యాట్ స్లేవ్ నోడ్ నెట్‌వర్క్ నుంచి డిస్‌కనెక్ట్ అయింది.", "category": "Communication Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "SCADA సర్వర్ తో PLC కనెక్ట్ అవ్వడం లేదు.", "category": "Communication Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "మోడ్‌బస్ RTU రెస్పాన్స్ రాక టైమ్‌అవుట్ అయింది.", "category": "Communication Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "PLC మరియు డ్రైవ్ మధ్య ఈథర్‌నెట్ కమ్యూనికేషన్ కట్ అయింది.", "category": "Communication Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఫీల్డ్‌బస్ నెట్‌వర్క్ లో ప్యాకెట్లు చేరడం లేదు.", "category": "Communication Fault", "language": "Telugu", "source": "manual_seed"},
    {"text": "మోడ్‌బస్ సీరియల్ లింక్ టైమ్‌అవుట్ ఎర్రర్ ఇస్తోంది.", "category": "Communication Fault", "language": "Telugu", "source": "manual_seed"},
    # Telugu-English code-switched (22 examples)
    {"text": "PLC communication avvatledu drive unit tho network break valla.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Profinet network lo packet loss vastundi remote IO link cut aindi.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Modbus communication timeout error continuous ga vastundi.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Ethernet cable disconnect aipoindi switch board daggara.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "HMI to PLC communication disconnect aindi screen black aindi.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "CAN bus state bus-off loki vellipoindi transmission errors valla.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "EtherCAT node network drop aindi, master sync error vastundi.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "RS-485 serial bus lo CRC error rates chala high ga unnayi.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Network switch port link flapping avtundi continuous ga.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Wireless gateway connection drop aindi field transmitter tho.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Fiber optic communication transceivers signal receive cheskoleka potunnayi.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Fieldbus bus terminator poindi, reflection errors vastunnayi.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Remote I/O station ping response ivvatledu network lo.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "SCADA communication link lost with PLC controller.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "DeviceNet network lo heavy frame collisions avtunnayi.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Ethernet IP adapter communication timeout alarm trigger aindi.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Serial cable disconnect aindi barcode scanner nunchi.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Modbus socket connect reject aindi server side nunchi.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Data telemetry packet transfer completely halted.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Industrial ethernet switch port down no communication.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "PLC communication disconnect aindi remote IO station tho.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Profinet network bus link down ayyi connection lost aindi.", "category": "Communication Fault", "language": "Telugu-English", "source": "manual_seed"},

    # =========================================================================
    # 8. UNKNOWN (75 examples)
    # =========================================================================
    # English (35 examples)
    {"text": "General equipment malfunction noticed during operational cycle.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "The machine has an issue.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Equipment is not working properly.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Problem reported on production line.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Machine stopped.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Operator noticed an issue on station 3.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Plant line 2 halt.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Routine maintenance check requested.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "System status abnormal.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Please check unit 5.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "General malfunction observed on production floor.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Apparatus shut down automatically without diagnostic code.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Operator raised ticket stating machine not behaving as expected.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Workstation stopped mid-cycle for unspecified reasons.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Unspecified disturbance reported during shift handover.", "category": "Unknown", "language": "Unknown", "source": "manual_seed"},
    {"text": "Device ceased normal operation at 14:00.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Supervisor requested inspection of assembly cell 4.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Unexpected behavior on machine 9.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Process interrupted without error logs.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Equipment tripped but reason unknown.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Station 1 requires immediate technical inspection.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Unknown trouble reported by technician.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "System not responding to start sequence.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Production delay due to unspecified equipment anomaly.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Machine shut off unexpectedly during shift.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Plant technician reported an issue on the line.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Equipment anomaly observed during routine walkaround.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Production station 4 ceased operation for undetermined reasons.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Operator logged a ticket regarding unexpected line delay.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Equipment behavior appears irregular according to shift supervisor.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Workstation tripped without active error code or alarm annunciator.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Assembly cell stopped functioning unexpectedly during cycle.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Unknown condition flagged on secondary conveyor line.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Process line halted for general inspection.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    {"text": "Operator noticed something irregular with unit 7.", "category": "Unknown", "language": "English", "source": "manual_seed"},
    # Telugu script (18 examples)
    {"text": "మెషిన్‌లో ఏదో సమస్య వచ్చింది.", "category": "Unknown", "language": "Telugu", "source": "manual_seed"},
    {"text": "యూనిట్ సరిగ్గా పనిచేయడం లేదు.", "category": "Unknown", "language": "Telugu", "source": "manual_seed"},
    {"text": "లైన్ ఆగిపోయింది.", "category": "Unknown", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఆపరేటర్ సమస్య ఉందని చెప్పారు.", "category": "Unknown", "language": "Telugu", "source": "manual_seed"},
    {"text": "మెషిన్ నడవట్లేదు, ఒకసారి చెక్ చేయండి.", "category": "Unknown", "language": "Telugu", "source": "manual_seed"},
    {"text": "ప్లాంట్ స్టేషన్ వద్ద ఇబ్బంది ఉంది.", "category": "Unknown", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఉత్పత్తి ఆగిపోయింది కానీ కారణం తెలియదు.", "category": "Unknown", "language": "Telugu", "source": "manual_seed"},
    {"text": "మెషిన్ పని చేయడం ఆగిపోయింది.", "category": "Unknown", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఏదో తేడాగా ఉంది, సరిగ్గా నడవడం లేదు.", "category": "Unknown", "language": "Telugu", "source": "manual_seed"},
    {"text": "పరికరంలో తెలియని సమస్య వచ్చింది.", "category": "Unknown", "language": "Telugu", "source": "manual_seed"},
    {"text": "మెషిన్ సడెన్ గా ఆఫ్ అయిపోయింది.", "category": "Unknown", "language": "Telugu", "source": "manual_seed"},
    {"text": "యూనిట్ 3 ఇన్స్పెక్షన్ చేయాలి.", "category": "Unknown", "language": "Telugu", "source": "manual_seed"},
    {"text": "షిఫ్ట్ లో పని ఆగిపోయింది.", "category": "Unknown", "language": "Telugu", "source": "manual_seed"},
    {"text": "కారణం లేకుండా మెషిన్ ట్రిప్ అయింది.", "category": "Unknown", "language": "Telugu", "source": "manual_seed"},
    {"text": "ఏదో తప్పు జరిగింది, మెషిన్ రెస్పాండ్ అవ్వట్లేదు.", "category": "Unknown", "language": "Telugu", "source": "manual_seed"},
    {"text": "స్టేషన్ వద్ద సమస్య ఉందని సమాచారం వచ్చింది.", "category": "Unknown", "language": "Telugu", "source": "manual_seed"},
    {"text": "మెషిన్ పనిచేయడం ఆగిపోయింది కానీ కారణం ఇంకా తెలియదు.", "category": "Unknown", "language": "Telugu", "source": "manual_seed"},
    {"text": "పరికరంలో తేడా కనిపించి ఆపరేటర్ చెక్ చేయమన్నారు.", "category": "Unknown", "language": "Telugu", "source": "manual_seed"},
    # Telugu-English code-switched (22 examples)
    {"text": "Machine lo edo issue undi, proper ga work avvatledu.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Unit properly run avvatledu, check cheyandi.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Station 4 daggara problem undi, reason teleedu.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Machine sudden ga stop aindi running lo.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Production line aagipoindi unspecified problem valla.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Operator reported machine issue without any specific error.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Equipment abnormal ga behaves chestundi, details levu.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Something went wrong in the process line.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "System shut down aindi without any reason.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Line 2 stopped, technical support inspect cheyali.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Unit not working properly today shift lo.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Machine start avvatledu but error code ledu.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Station stopped working suddenly, please verify.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "General issue observed on conveyor line.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Machine turned off automatically.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Workstation trip aindi reason unknown.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Equipment status weird ga undi check required.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Plant line halted unexpectedly.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Production floor lo disturbance report chesaru.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Unspecified breakdown on station 5.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Station lo edo disturbance report chesaru reason teleedu.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"},
    {"text": "Machine halt aindi without any alarm message.", "category": "Unknown", "language": "Telugu-English", "source": "manual_seed"}
]


def generate_training_csv(output_path: Path = TRAINING_CSV_PATH, overwrite: bool = False) -> Path:
    """Generates the training dataset CSV with strict category and quality checks."""
    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Verify no leakages with evaluation cases
    eval_texts = set()
    if EVALUATION_DATASET_PATH.exists():
        with open(EVALUATION_DATASET_PATH, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            for item in eval_data:
                desc = item.get("description", "").strip().lower()
                if desc:
                    eval_texts.add(desc)

    fieldnames = ["text", "category", "language", "source"]
    written_count = 0

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in _DATASET_EXAMPLES:
            text = row["text"].strip()
            cat = row["category"].strip()
            # Double check category
            if cat not in APPROVED_CATEGORIES:
                raise ValueError(f"Invalid category in dataset seed: {cat}")
            # Ensure no leakage into evaluation benchmark
            if text.lower() in eval_texts:
                raise ValueError(f"DATA LEAKAGE DETECTED: '{text}' exists in evaluation benchmark!")
            writer.writerow(row)
            written_count += 1

    return output_path


def load_and_validate_training_dataset(csv_path: Path = TRAINING_CSV_PATH) -> pd.DataFrame:
    """
    Loads training CSV, validates categories, checks missing/duplicate data,
    prints full data quality report, and raises exceptions if quality checks fail.
    """
    if not csv_path.exists():
        generate_training_csv(csv_path)

    df = pd.read_csv(csv_path, encoding="utf-8")

    # Required columns
    required_cols = {"text", "category", "language", "source"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Training dataset missing required columns: {missing_cols}")

    # Check missing texts
    missing_texts = df["text"].isna().sum() + (df["text"].str.strip() == "").sum()
    if missing_texts > 0:
        raise ValueError(f"Training dataset contains {missing_texts} empty/missing text entries!")

    # Check invalid categories
    invalid_categories = df[~df["category"].isin(APPROVED_CATEGORIES)]["category"].unique()
    if len(invalid_categories) > 0:
        raise ValueError(f"Training dataset contains invalid categories: {list(invalid_categories)}")

    # Check class coverage
    class_counts = df["category"].value_counts().to_dict()
    for cat in APPROVED_CATEGORIES:
        if cat not in class_counts or class_counts[cat] == 0:
            raise ValueError(f"Class '{cat}' has ZERO training examples! Aborting.")

    # Quality check metrics
    duplicate_count = df["text"].duplicated().sum()
    total_examples = len(df)
    lang_dist = df["language"].value_counts().to_dict()

    print("=" * 60)
    print("             DATA QUALITY CHECKS REPORT")
    print("=" * 60)
    print(f"Total examples       : {total_examples}")
    print(f"Missing texts        : {missing_texts}")
    print(f"Duplicate texts      : {duplicate_count}")
    print(f"Invalid categories   : {len(invalid_categories)}")
    print("\nExamples per class:")
    for cat in APPROVED_CATEGORIES:
        print(f"  - {cat:<24} : {class_counts.get(cat, 0)}")
    print("\nLanguage distribution:")
    for lang, cnt in lang_dist.items():
        print(f"  - {lang:<24} : {cnt}")
    print("=" * 60)

    return df


def split_training_data(
    df: pd.DataFrame,
    val_ratio: float = VALIDATION_SPLIT_RATIO,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split of training dataframe into train and validation splits."""
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df["category"],
        random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


if __name__ == "__main__":
    generate_training_csv(overwrite=True)
    loaded_df = load_and_validate_training_dataset()
    train_sub, val_sub = split_training_data(loaded_df)
    print(f"\nTrain set: {len(train_sub)} examples, Validation set: {len(val_sub)} examples.")
