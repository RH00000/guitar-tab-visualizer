from src.parser import parse_tab

with open("data/hotel_california_solo.txt") as f:
    raw = f.read()

notes = parse_tab(raw)
notes.sort(key=lambda n: n.column)
for note in notes:
    print(note)