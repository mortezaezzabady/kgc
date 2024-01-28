
def clean_data(filename):
    data = []
    with open('data/wn18rr/raw/' + filename + '.txt', 'r', encoding='utf-8') as f:
        for line in f:
            head, rel, tail = line.strip().split('\t')
            head = head.split('.')[0]
            rel = rel[1:]
            tail = tail.split('.')[0]
            # head = ' '.join(head.split('.')[0].split('_'))
            # rel = ' '.join(rel[1:].split('_'))
            # tail = ' '.join(tail.split('.')[0].split('_'))
            data.append('\t'.join([head, rel, tail]))
    with open('data/wn18rr/' + filename + '.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(data))

# clean_data('train')
# clean_data('valid')
# clean_data('test')
        
entities = set()
with open('data/wn18rr/entity2id.txt', 'r', encoding='utf-8') as f:
    for line in f:
        entities.add(line.strip().split('\t')[0])
print(len(entities))
cnt = 0
tot = 0
data = []
with open('data/wn18rr/valid.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    tot = len(lines)
    for line in lines:
        head, rel, tail = line.strip().split('\t')
        if head not in entities:
            print(head)
            continue
        if tail not in entities:
            print(tail)
            continue
        cnt += 1
        data.append('\t'.join([head, rel, tail]))
with open('data/wn18rr/valid1.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(data))