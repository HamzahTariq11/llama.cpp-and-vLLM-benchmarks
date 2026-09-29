"""Generate bench/prompts.jsonl. All text is original to this repo.

Usage:
    uv run python bench/make_prompts.py bench/prompts.jsonl
"""

import json
import sys

SHORT = [
    "What is the capital of Australia?",
    "Give me three synonyms for 'quick'.",
    "Convert 72 degrees Fahrenheit to Celsius and show the formula.",
    "What does HTTP status code 404 mean?",
    "Name the planets of the solar system in order from the sun.",
    "What is the difference between a list and a tuple in Python?",
    "Write a haiku about autumn rain.",
    "Explain what a prime number is in one sentence.",
    "What year did the first human land on the Moon?",
    "Translate 'good morning, how are you?' into Spanish.",
    "What is 17 multiplied by 23?",
    "Suggest a name for a small coffee shop near a library.",
]

MEDIUM = [
    "Explain how a hash table works, including what happens on a collision.",
    "Write a short story (about 150 words) about a lighthouse keeper who finds a message in a bottle.",
    "Compare TCP and UDP. When would you choose each one?",
    "Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation, with a docstring and two example calls.",
    "Describe the water cycle to a ten-year-old.",
    "What are the pros and cons of working from home? Give at least three of each.",
    "Explain the difference between supervised and unsupervised learning with one example of each.",
    "Write a polite email declining a meeting invitation and proposing two alternative times.",
    "How does compound interest work? Include a worked example with numbers.",
    "Give step-by-step instructions for making a basic vegetable soup.",
    "Explain what a SQL JOIN is and describe INNER, LEFT and FULL OUTER joins.",
    "Write a limerick about a cat who learns to code.",
    "What is recursion? Show a simple recursive function for factorial in Python and explain the base case.",
    "Summarize the main causes of inflation in plain language.",
    "Plan a three-day itinerary for a first-time visitor to a mid-sized coastal city.",
    "Explain Big-O notation and give the complexity of binary search and bubble sort.",
]

PASSAGES = [
    (
        "The town council met on Tuesday evening to discuss the proposal for a new bicycle lane "
        "along Mill Street. Supporters argued that the lane would reduce traffic, improve safety for "
        "children cycling to school, and encourage local businesses by bringing more foot traffic to "
        "the street. Opponents, many of them shop owners, worried that removing twenty parking spaces "
        "would drive away customers who arrive by car, especially older residents. The traffic "
        "engineer presented data showing that on-street parking was, on average, only sixty percent "
        "occupied during business hours, and that a similar lane on Station Road had coincided with a "
        "small increase in retail spending. After two hours of debate, the council voted five to two "
        "to run a six-month trial, with a promise to review parking and sales figures before making "
        "the lane permanent."
    ),
    (
        "Sourdough bread relies on a starter: a living culture of wild yeast and lactic acid "
        "bacteria kept in a mixture of flour and water. The yeast produces carbon dioxide, which "
        "makes the dough rise, while the bacteria produce acids that give the bread its tangy flavor "
        "and help it keep longer. Bakers feed the starter regularly by discarding part of it and "
        "adding fresh flour and water. A healthy starter roughly doubles in size a few hours after "
        "feeding. Because wild yeast works more slowly than commercial yeast, sourdough often needs a "
        "long, cool fermentation, sometimes overnight in a refrigerator. This slow process develops "
        "flavor and makes the crumb more open, but it also makes timing less predictable, which is "
        "why many home bakers keep notes on temperature and rise times."
    ),
    (
        "Our team moved the nightly reporting job from a single large server to a set of small "
        "workers pulling tasks from a queue. Before the change, the job took about four hours and "
        "failed completely if any step crashed, forcing a full rerun the next morning. After the "
        "change, each report is an independent task; failed tasks are retried up to three times and "
        "then sent to a dead-letter queue for a human to inspect. Total runtime dropped to about "
        "fifty minutes with eight workers. However, we discovered two new problems: some reports "
        "depended on others and occasionally ran before their inputs were ready, and the database "
        "saw bursts of load when all workers started at once. We fixed the first by adding explicit "
        "dependencies between tasks and the second by staggering worker start times and adding a "
        "connection pool limit."
    ),
    (
        "Honeybees communicate the location of food through a behavior known as the waggle dance. "
        "A forager returning to the hive walks in a figure-eight pattern, and during the straight "
        "middle section she waggles her body from side to side. The angle of this straight run, "
        "relative to vertical on the comb, indicates the direction of the food relative to the sun. "
        "The duration of the waggle indicates distance: longer waggles mean the food is farther away. "
        "Other bees follow the dancer, pick up the scent of the flowers she visited, and then fly out "
        "to find the source. Because the sun moves across the sky during the day, bees adjust the "
        "angle of their dances over time, which suggests they have an internal sense of time."
    ),
]

LONG = [
    f"Summarize the following text in three bullet points.\n\n{PASSAGES[0]}",
    f"Read the passage and answer: what are two downsides of the approach described, and how might they be addressed?\n\n{PASSAGES[0]}",
    f"Explain the following passage to someone who has never baked, then list the three most important tips it implies.\n\n{PASSAGES[1]}",
    f"Rewrite the following passage as a short, friendly blog post intro of about 100 words.\n\n{PASSAGES[1]}",
    f"Write a short incident-style postmortem (summary, impact, root causes, fixes) based on this description.\n\n{PASSAGES[2]}",
    f"List every technical change mentioned below and the problem each one solved.\n\n{PASSAGES[2]}",
    f"Create five quiz questions with answers based on this passage.\n\n{PASSAGES[3]}",
    f"Summarize this passage in one paragraph, then explain what the last sentence implies.\n\n{PASSAGES[3]}",
    f"Compare the two passages below: what do they have in common in how a system adapts over time?\n\nPassage A:\n{PASSAGES[1]}\n\nPassage B:\n{PASSAGES[3]}",
    f"Using both passages below, write a short paragraph on tradeoffs between speed and reliability.\n\nPassage A:\n{PASSAGES[0]}\n\nPassage B:\n{PASSAGES[2]}",
    f"Extract all numbers mentioned in the passage and explain what each refers to.\n\n{PASSAGES[0]}\n\n{PASSAGES[2]}",
    f"Write a critical review of the decision-making described below, noting any missing information.\n\n{PASSAGES[0]}",
]

# ~1.8k-token shared system document for the prefix-caching test (fictional organization).
HANDBOOK = """You are the help assistant for the Harbor Lane Tool Library, a volunteer-run lending library for tools and equipment. Answer member questions using only the handbook below. If the handbook does not cover a question, say so and suggest contacting a volunteer coordinator. Keep answers short and friendly.

HARBOR LANE TOOL LIBRARY - MEMBER HANDBOOK

1. About the library
The Harbor Lane Tool Library lends tools, garden equipment, kitchen appliances and camping gear to members of the community. It is run entirely by volunteers and funded by membership fees, small donations and an annual fundraising fair. The library occupies the former boathouse at the end of Harbor Lane, next to the public slipway. Our aim is to reduce waste, save members money, and help people learn practical skills.

2. Opening hours
The library is open Tuesday and Thursday from 5:00 pm to 8:00 pm, and Saturday from 9:00 am to 1:00 pm. It is closed on public holidays and for two weeks over the winter break, usually the last week of December and the first week of January. Opening hours may change during the annual fair in June; changes are posted on the noticeboard by the front door and in the monthly newsletter.

3. Membership
Membership is open to anyone aged 18 or over who lives, works or studies within the district. There are three membership levels:
- Basic: 25 per year. Borrow up to 3 items at a time.
- Household: 40 per year. Up to 2 adults at the same address; borrow up to 5 items at a time between them.
- Supporter: 60 per year. Same as Household, plus priority booking for high-demand items and a 10 percent discount at workshops.
Concession rates of half the listed price are available for students, people over 65 and anyone receiving income support. To join, bring photo identification and proof of address to any opening session. Members must read and sign the safety agreement before their first loan. Memberships run for twelve months from the date of joining and are not refundable.

4. Borrowing
The standard loan period is 7 days. Items can be renewed once for a further 7 days, provided no other member has reserved the item. Renewals can be made in person or by replying to the reminder email sent two days before the due date. High-demand items, marked with a red tag, have a loan period of 4 days and cannot be renewed. These currently include the carpet cleaner, the pressure washer, the tile cutter, the large dehumidifier and the pop-up gazebo.
Members can reserve up to two items in advance through the online catalogue. Reservations are held for 24 hours after the item becomes available; after that, the reservation passes to the next person on the waiting list.

5. Returning items
Items must be returned clean and in the same condition as when borrowed, with all parts, attachments and manuals. Return items during opening hours to the returns desk; do not leave items outside the building. Batteries for cordless tools should be returned charged if possible, but a flat battery is not penalised. Garden tools must be free of soil. Kitchen appliances must be washed and completely dry.

6. Late returns and fees
Late items are charged at 1 per item per day, up to a maximum of 15 per item. Red-tag items are charged at 3 per day, up to a maximum of 30. Members with outstanding fees cannot borrow new items until fees are paid. Fees can be waived by a volunteer coordinator in cases of illness, bereavement or other genuine hardship; please speak to us rather than staying away.
If an item is lost, the member pays the replacement cost listed in the catalogue, minus any late fees already paid for that item. If an item is damaged through normal use, there is no charge, but please tell the returns desk so we can repair it. If an item is damaged through misuse, the member may be asked to contribute up to half the repair cost.

7. Safety
Every power tool in the catalogue has a safety rating from 1 to 3.
- Rating 1: No induction needed (for example, cordless drills, sanders, hedge trimmers).
- Rating 2: Members must watch a short safety video and pass a five-question quiz at the desk before first borrowing (for example, jigsaws, circular saws, the pressure washer).
- Rating 3: Members must attend an in-person induction led by a qualified volunteer (for example, the chainsaw, the tile cutter, the petrol lawn mower). Inductions run on the first Saturday of each month and must be booked in advance. Rating 3 items cannot be borrowed by members under 21.
Protective equipment such as goggles, ear defenders and gloves is lent free of charge with any rated tool. Members are responsible for using tools safely and following the manufacturer's instructions. The library is not liable for injuries caused by misuse.

8. Workshops
The library runs regular workshops, including basic home repairs, bike maintenance, sewing machine basics, knife sharpening and garden pruning. Workshops cost 10 for members and 15 for non-members, with a maximum of eight participants. Supporter members receive a 10 percent discount. Booking is required through the online calendar; cancellations made at least 48 hours before the workshop are refunded in full.

9. Repair café
On the last Saturday of each month, the library hosts a repair café from 10:00 am to 1:00 pm. Anyone, member or not, can bring a broken household item for volunteers to help fix: small appliances, clothing, bicycles, toys and furniture. Repairs are free, but donations are welcome. Volunteers cannot repair gas appliances, microwaves, or anything involving mains wiring inside walls. Items are repaired with the owner present, so that people learn how to do it themselves next time.

10. Donations
We welcome donations of good-quality tools and equipment in working order. Please check the wish list on the noticeboard before bringing large items. We cannot accept petrol equipment older than ten years, anything with missing safety guards, mattresses, or upholstered furniture. Donated items that do not suit the catalogue may be sold at the annual fair to raise funds.

11. Volunteering
Volunteers staff the desk, check and repair returned items, run workshops, and help at the fair. Volunteers who give at least six hours a month receive free Basic membership. New volunteers attend a two-hour orientation and shadow an experienced volunteer for their first two sessions.

12. Contact and feedback
Questions can be asked in person during opening hours or by leaving a note in the feedback box by the entrance. Complaints are reviewed by the volunteer coordinators at their monthly meeting, and we aim to reply within 14 days. Lost property is kept for one month and then donated."""

SHARED_QUESTIONS = [
    "What are the opening hours on Saturday?",
    "How much does a Household membership cost, and how many items can we borrow?",
    "I'm a student. How much would Basic membership cost me?",
    "Can I renew the pressure washer?",
    "My drill is three days late. How much do I owe?",
    "What do I need to do before borrowing a circular saw?",
    "I'm 19. Can I borrow the chainsaw?",
    "Can I bring my broken microwave to the repair café?",
    "How long are reservations held once an item becomes available?",
    "I damaged a sander during normal use. Will I be charged?",
    "How do I become a volunteer and what do I get for it?",
    "Will you accept a 15-year-old petrol lawn mower as a donation?",
]


def main(out: str) -> None:
    records = []
    for length, prompts in (("short", SHORT), ("medium", MEDIUM), ("long", LONG)):
        for i, text in enumerate(prompts, 1):
            records.append({"id": f"{length}-{i:02d}", "set": "varied", "prompt": text})
    records.append({"id": "shared-system", "set": "shared_prefix_system", "prompt": HANDBOOK})
    for i, text in enumerate(SHARED_QUESTIONS, 1):
        records.append({"id": f"shared-{i:02d}", "set": "shared_prefix", "prompt": text})
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    varied = sum(r["set"] == "varied" for r in records)
    print(f"wrote {len(records)} records: {varied} varied, {len(SHARED_QUESTIONS)} shared-prefix")


if __name__ == "__main__":
    main(sys.argv[1])
