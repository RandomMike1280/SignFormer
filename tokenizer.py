import pickle

class CharacterLevelTokenizer:
    def __init__(self, chars=""):
        self.chars = ['<|gaytokennotfound|>'] + sorted(list(set(chars)))
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    def encode(self, s):
        tokens = []
        for ch in s:
            if ch in self.char_to_int:
                tokens.append(self.char_to_int[ch])
            else:
                tokens.append(self.char_to_int['<|gaytokennotfound|>'])
        return tokens

    def decode(self, l):
        return ''.join([self.int_to_char[i] for i in l])
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

if __name__ == "__main__":
    with open("dataset.txt", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharacterLevelTokenizer(text)
    tokenizer.save("tokenizer.pkl")
    tokenizer = CharacterLevelTokenizer.load("tokenizer.pkl")
    print(tokenizer.encode("hii there"))
    print(tokenizer.decode(tokenizer.encode("hii there")))
    print(tokenizer.chars)
