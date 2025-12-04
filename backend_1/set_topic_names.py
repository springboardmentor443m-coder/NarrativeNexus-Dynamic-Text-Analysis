import pickle
import os

# Load keywords
with open("backend_1/models/topic_keywords.pkl", "rb") as f:
    keywords = pickle.load(f)

# ---- Edit this dict and put your own names ----
topic_names = {
    0: "Sports",
    1: "Iraq / Iran / Middle Eastern Countries",
    2: "Oil Prices / OPEC",
    3: "Middle East Conflict",
    4: "Airways / Airlines",
    5: "Agreements / Bonds",
    6: "Banks / Funds",
    7: "US Politics",
    8: "Space / Interstellar Exploration",
    9: "Malware / Cyber Attacks",
    10: "Eastern Countries Talks",
    11: "Climate Changes / Environmental Issues",
    12: "Georgia Related",
    13: "Gaming",
    14: "Natural Disasters",
    15: "Labour / Working People's hurdles",
    16: "Diseases / Flu",
    17: "Medical Drugs",
    18: "Genetics / Cloning Experiments",
    19: "Oil, Taxes & Bills",
    20: "carter, manning, quincy, jets, eli, dolphins, bears, quarterback, ogunleye, nfl",
    21: "Army / Troops / Wars",
    22: "Business People / CEOs / Franchises",
    23: "Nepal's Extremists",
    24: "Crime / Investigation / Networks",
    25: "Crime Mob / Mafia",
    26: "European Segments",
    27: "Power Source / Batteries",
    28: "Chess / Championship",
    29: "Philippines / Crisis",
    30: "Museum / Archives",
    31: "hicks, howard, australian, overboard, scrafton, father, crimes, john, tribunal, david",
    32: "Knowlege / Language / Maths",
    33: "Diet & Health",
    34: "Archaeology & Exploration",
    35: "Not Found",
    36: "Loans & Mortgage",
    37: "schroeder, russian, chancellor, gerhard, girl, zoo, adopted, wife, german, germany",
    38: "selig, 2009, baseball, bud, extension, owners, commissioner, contract, through, extend",
    # Add all 40 topics here
}
# ------------------------------------------------

# Save them
with open("backend_1/models/topic_names.pkl", "wb") as f:
    pickle.dump(topic_names, f)

print("Custom topic names saved!")
