import torch.nn as nn
import layers


class Seq2Seq(nn.Module):
    def __init__(self, config):
        super(Seq2Seq, self).__init__()

        self.config = config
        self.encoder = layers.EncoderRNN(config.vocab_size,
                                         config.embedding_size,
                                         config.encoder_hidden_size,
                                         config.rnn,
                                         config.num_layers,
                                         False,
                                         config.dropout)

        self.decoder = layers.DecoderRNN(config.vocab_size,
                                         config.embedding_size,
                                         config.decoder_hidden_size,
                                         config.rnncell,
                                         config.num_layers,
                                         config.dropout,
                                         config.word_drop,
                                         config.max_unroll,
                                         config.sample,
                                         config.temperature,
                                         config.beam_size)

        if config.tie_embedding:
            self.decoder.embedding = self.encoder.embedding

    def forward(self, input_sentences, input_sentence_length,
                input_conversation_length, target_sentences,
                decode=False, return_encoder_output=False):
        """
        Args:
            input_sentences: (Variable, LongTensor) [num_sentences, seq_len]
            target_sentences: (Variable, LongTensor) [num_sentences, seq_len]
        Return:
            decoder_outputs: (Variable, FloatTensor)
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """
        num_sentences = input_sentences.size(0)

        # encoder_outputs: [num_sentences, max_source_length, hidden_size * direction]
        # encoder_hidden: [num_layers * direction, num_sentences, hidden_size]
        encoder_outputs, encoder_hidden = self.encoder(input_sentences,
                                                       input_sentence_length)

        # encoder_hidden: [num_sentences, num_layers * direction * hidden_size]
        encoder_hidden = encoder_hidden.transpose(
            1, 0).contiguous().view(num_sentences, -1)

        decoder_init = encoder_hidden

        # [num_layers, batch_size, hidden_size]
        decoder_init = decoder_init.view(
            self.decoder.num_layers, -1, self.decoder.hidden_size)

        if not decode:
            # [batch_size, seq_len, vocab_size]
            decoder_outputs = self.decoder(target_sentences,
                                           init_h=decoder_init,
                                           decode=decode)

            if return_encoder_output:
                return encoder_outputs, decoder_outputs
            else:
                return decoder_outputs
        else:
            # decoder_outputs = self.decoder(target_sentences,
            #                                init_h=decoder_init,
            #                                decode=decode)
            # return decoder_outputs.unsqueeze(1)
            # prediction: [batch_size, beam_size, max_unroll]
            prediction, final_score, length = self.decoder.beam_decode(
                init_h=decoder_init)

            # Get top prediction only
            # [batch_size, max_unroll]
            # prediction = prediction[:, 0]

            # [batch_size, beam_size, max_unroll]
            return prediction
