import torch.nn as nn
import torch
import layers
from utils import to_var, pad
from seq2seq import Seq2Seq


class AEMapper(nn.Module):
    def __init__(self, config):
        super(AEMapper, self).__init__()

        self.config = config
        self.source_seq2seq = Seq2Seq(config)

        self.target_seq2seq = Seq2Seq(config)

        self.mapper = layers.FeedForward(config.context_size,
                                         config.num_layers * config.decoder_hidden_size,
                                         num_layers=1,
                                         activation=config.activation)

    def forward(self, input_sentences, input_sentence_length,
                input_conversation_length, target_sentences, decode=False):
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
        max_len = input_conversation_length.data.max().item()

        # encoder_outputs: [num_sentences, max_source_length, hidden_size * direction]
        # encoder_hidden: [num_layers * direction, num_sentences, hidden_size]
        encoder_outputs, encoder_hidden = self.source_encoder(input_sentences,
                                                              input_sentence_length)

        # encoder_hidden: [num_sentences, num_layers * direction * hidden_size]
        encoder_hidden = encoder_hidden.transpose(
            1, 0).contiguous().view(num_sentences, -1)

        # pad and pack encoder_hidden
        start = torch.cumsum(torch.cat((to_var(input_conversation_length.data.new(1).zero_()),
                                        input_conversation_length[:-1])), 0)

        # encoder_hidden: [batch_size, max_len, num_layers * direction * hidden_size]
        encoder_hidden = torch.stack([pad(encoder_hidden.narrow(0, s, l), max_len)
                                      for s, l in zip(start.data.tolist(),
                                                      input_conversation_length.data.tolist())], 0)
        encoder_hidden = encoder_hidden[:, -1, :]

        # project context_outputs to decoder init state
        decoder_init = self.mapper(encoder_hidden)

        # [num_layers, batch_size, hidden_size]
        decoder_init = decoder_init.view(
            self.source_decoder.num_layers, -1, self.source_decoder.hidden_size)

        # train: [batch_size, seq_len, vocab_size]
        # eval: [batch_size, seq_len]
        if not decode:

            decoder_outputs = self.source_decoder(target_sentences,
                                                  init_h=decoder_init,
                                                  decode=decode)
            return decoder_outputs

        else:
            # decoder_outputs = self.source_decoder(target_sentences,
            #                                init_h=decoder_init,
            #                                decode=decode)
            # return decoder_outputs.unsqueeze(1)
            # prediction: [batch_size, beam_size, max_unroll]
            prediction, final_score, length = self.source_decoder.beam_decode(
                init_h=decoder_init)

            # Get top prediction only
            # [batch_size, max_unroll]
            # prediction = prediction[:, 0]

            # [batch_size, beam_size, max_unroll]
            return prediction

    def generate(self, context, sentence_length, n_context):
        # context: [batch_size, n_context, seq_len]
        batch_size = context.size(0)
        # n_context = context.size(1)
        samples = []

        # Run for context
        context_hidden = None
        for i in range(n_context):
            # encoder_outputs: [batch_size, seq_len, hidden_size * direction]
            # encoder_hidden: [num_layers * direction, batch_size, hidden_size]
            encoder_outputs, encoder_hidden = self.source_encoder(context[:, i, :],
                                                                  sentence_length[:, i])

            encoder_hidden = encoder_hidden.transpose(
                1, 0).contiguous().view(batch_size, -1)
            # context_outputs: [batch_size, 1, context_hidden_size * direction]
            # context_hidden: [num_layers * direction, batch_size, context_hidden_size]
            context_outputs, context_hidden = self.context_encoder.step(encoder_hidden,
                                                                        context_hidden)

        # Run for generation
        for j in range(self.config.n_sample_step):
            # context_outputs: [batch_size, context_hidden_size * direction]
            context_outputs = context_outputs.squeeze(1)
            decoder_init = self.context2decoder(context_outputs)
            decoder_init = decoder_init.view(
                self.source_decoder.num_layers, -1, self.source_decoder.hidden_size)

            prediction, final_score, length = self.source_decoder.beam_decode(
                init_h=decoder_init)
            # prediction: [batch_size, seq_len]
            prediction = prediction[:, 0, :]
            # length: [batch_size]
            length = [l[0] for l in length]
            length = to_var(torch.LongTensor(length))
            samples.append(prediction)

            encoder_outputs, encoder_hidden = self.source_encoder(prediction,
                                                                  length)

            encoder_hidden = encoder_hidden.transpose(
                1, 0).contiguous().view(batch_size, -1)

            context_outputs, context_hidden = self.context_encoder.step(encoder_hidden,
                                                                        context_hidden)

        samples = torch.stack(samples, 1)
        return samples
