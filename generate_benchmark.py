import json
import random


domains = {


"programming_language":[
("Python","Java"),
("Java","Python"),
("C++","Python")
],


"framework":[
("FastAPI","Flask"),
("React","Angular")
],


"database":[
("PostgreSQL","MongoDB"),
("MySQL","SQLite")
],


"food":[
("vegetarian","non vegetarian"),
("vegan","meat")
],


"location":[
("Hyderabad","Bangalore"),
("India","USA")
],


"theme":[
("dark mode","light mode")
]

}



cases=[]


for i in range(500):

    category=random.choice(list(domains.keys()))

    old_value,new_value=random.choice(
        domains[category]
    )


    case_type=random.choice(
        [
            "conflict",
            "duplicate",
            "different_attribute"
        ]
    )


    if case_type=="duplicate":

        new_value=old_value
        expected="SAFE"


    elif case_type=="conflict":

        expected="UNSAFE"


    else:

        expected="SAFE"



    old={
        "subject":"user",
        "attribute":category,
        "value":old_value,
        "scope":"general",
        "context":"general"
    }


    new={
        "subject":"user",
        "attribute":category if case_type!="different_attribute" else "other",
        "value":new_value,
        "scope":"general",
        "context":"general"
    }


    cases.append(
        {
        "old":old,
        "new":new,
        "expected":expected
        }
    )



with open(
"data/safety_gate_benchmark.json",
"w"
) as f:

    json.dump(
        cases,
        f,
        indent=4
    )


print("Generated",len(cases),"cases")