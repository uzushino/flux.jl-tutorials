mutable struct Lang
  name :: String
  word2index :: Dict{String, Int64}
  word2count :: Dict{String, Int64}
  index2word :: Dict{Int64, String}
  n_words :: Int64

  Lang(name) = new(name, Dict(), Dict(), Dict(1 => "SOS", 2 => "EOS"), 3)

end # struct

function addSentence(self::Lang, sentence::String)
  foreach(split(sentence, " ")) do word
    addWord(self, string(word))
  end
end

function addWord(self::Lang, word::String) 
  if !haskey(self.word2index, word)
    self.word2index[word] = self.n_words
    self.word2count[word] = 1
    self.index2word[self.n_words] = word
    self.n_words += 1
  else
    self.word2count[word] += 1
  end
end