import json
import random


templates = [

# duplicates
(
"I prefer dark mode",
"Application uses dark mode normally",
"Preserve"
),

(
"ACMA backend uses FastAPI",
"ACMA backend uses FastAPI",
"Ignore"
),


# updates
(
"ACMA backend uses FastAPI",
"I switched ACMA backend from FastAPI to Flask",
"Resolve"
),

(
"User prefers email notifications",
"I now prefer WhatsApp notifications",
"Resolve"
),


# events
(
"User is vegetarian",
"I ate chicken at a wedding",
"Preserve"
),


(
"Application normally uses dark mode",
"For tomorrow presentation use light mode",
"Preserve"
),


# different contexts
(
"Java is used for DSA coursework",
"Python is used for ACMA project",
"Preserve"
),


# deadlines
(
"ACMA paper deadline is September 20",
"ACMA paper deadline moved to September 27",
"Resolve"
),


# location
(
"User lives in Hyderabad",
"User travelled to Bangalore for conference",
"Preserve"
)

]


cases=[]


for i in range(500):

    old,new,label=random.choice(templates)

    cases.append(
        {
            "old":old,
            "new":new,
            "expected":label
        }
    )


with open(
"data/full_pipeline_benchmark.json",
"w"
) as f:

    json.dump(
        cases,
        f,
        indent=4
    )


print("Generated",len(cases),"cases")