#coding:utf-8
import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet152(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        net = self.resnet[:-1]
        net_last = self.resnet[-1]
        net_pre = net_last[:-1]
        """
        process the res5c_branch2a~2c
        """
        net_seg = net_last[-1]
        out = net(images)
        out = net_pre(out)
        temp = net_seg(out)
        """
        out = net_seg.conv1(out)#conv_res5c_branch2a(batch_size 512 8 8)
        out = net_seg.bn1(out)
        out = net_seg.relu(out)

        out = net_seg.conv2(out)#conv_res5c_branch2b(batch_size 512 8 8)
        out = net_seg.bn2(out)
        out = net_seg.relu(out)


        out = net_seg.conv3(out)#conv_res5c(batch_size 2048 8 8)
        out = net_seg.bn3(out)
        out = net_seg.relu(out)#relu_res5c
    

        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)

        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        """
        result = self.resnet(images)
        #result = self.adaptive_pool(result)
        result = result.permute(0, 2, 3, 1)




        return result, out


    def conv_res5c_2b(self, input):

        net_last = self.resnet[-1]
        """
        process the res5c_branch2b
        """
        net_seg = net_last[-1]
        out = net_seg.conv1(input)
        out = net_seg.bn1(out)
        out = net_seg.relu(out)
        out = net_seg.conv2(out)#conv_res5c_branch2b
        return out

    def conv_res5c_2c(self, input):
        net_last = self.resnet[-1]
        """
        process the res5c_branch2c
        """
        net_seg = net_last[-1]
        out = net_seg.bn2(input)
        out = net_seg.relu(out)
        out = net_seg.conv3(out)#conv_res5c
        return out

    def extra_layer(self, input):
        net_last = self.resnet[-1]
        net_seg = net_last[-1]
        out = net_seg.bn3(input)
        out = net_seg.relu(out)#relu_res5c
        #out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out


    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?

        """

        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune




class Attention_1(nn.Module):
    """
    这一部分是attention的内容
    """
    def __init__(self, encoder_dim, attention_dim):
        super(Attention_1, self).__init__()
        self.encoder = Encoder()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        """
        for channel attention
        """
        self.Wc = nn.Linear(64, 64)
        self.Whc = nn.Linear(512, 64)
        self.Wic = nn.Linear(64,1)
        '''
        for spatial attention
        '''
        self.Ws = nn.Linear(512, 1)
        self.Whs = nn.Linear(512, 64)
        self.Wis = nn.Linear(64, 64)
        """
        for next layer
        """
        self.NextlayerAttention = Attention_2()

    def forward(self, feature_map, decoder_hidden):
        batch_size = feature_map.size(0)
        feature_map = self.encoder.conv_res5c_2b(feature_map)
        V = feature_map.view(batch_size, 512, -1)
        b = self.tanh((self.Wc(V) + self.Wc.bias) + self.Whc(decoder_hidden).unsqueeze(1))
        beta = self.softmax(self.Wic(b) + self.Wic.bias)
        beta = beta.unsqueeze(2)
        attention_weighted_encoding = torch.mul(feature_map, beta)

        V = attention_weighted_encoding.view(batch_size, 512, -1)
        V = V.permute(0, 2, 1)
        a = self.tanh((self.Ws(V) + self.Ws.bias).squeeze(2) + self.Whs(decoder_hidden))
        alpha = self.softmax(self.Wis(a) + self.Wis.bias)
        alpha = alpha.reshape(batch_size, 8, 8).unsqueeze(1)

        attention_weighted_encoding = torch.mul(attention_weighted_encoding, alpha)

        attention_weighted_encoding = self.NextlayerAttention(attention_weighted_encoding, decoder_hidden)


        attention_weighted_encoding = self.encoder.extra_layer(attention_weighted_encoding)
        attention_weighted_encoding = attention_weighted_encoding.view(batch_size, -1, 2048).sum(dim=1)
        return attention_weighted_encoding

class Attention_2(nn.Module):
    """
    这一部分是attention的内容
    """
    def __init__(self):
        super(Attention_2, self).__init__()
        self.encoder = Encoder()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        """
        for channel attention
        """
        self.Wc = nn.Linear(64, 64)
        self.Whc = nn.Linear(512, 64)
        self.Wic = nn.Linear(64,1)
        '''
        for spatial attention
        '''
        self.Ws = nn.Linear(2048, 1)
        self.Whs = nn.Linear(512, 64)
        self.Wis = nn.Linear(64, 64)


    def forward(self, feature_map, decoder_hidden):
        batch_size = feature_map.size(0)
        feature_map = self.encoder.conv_res5c_2c(feature_map)
        V = feature_map.view(batch_size, 2048, -1)
        b = self.tanh((self.Wc(V) + self.Wc.bias) + self.Whc(decoder_hidden).unsqueeze(1))
        beta = self.softmax(self.Wic(b) + self.Wic.bias)
        beta = beta.unsqueeze(2)
        attention_weighted_encoding = torch.mul(feature_map, beta)

        V = attention_weighted_encoding.view(batch_size, 2048, -1)
        V = V.permute(0, 2, 1)
        a = self.tanh((self.Ws(V) + self.Ws.bias).squeeze(2) + self.Whs(decoder_hidden))
        alpha = self.softmax(self.Wis(a) + self.Wis.bias)
        alpha = alpha.reshape(batch_size, 8, 8).unsqueeze(1)

        attention_weighted_encoding = torch.mul(attention_weighted_encoding, alpha)


        return attention_weighted_encoding











class Decoder(nn.Module):
    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(Decoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.attention = Attention_1(encoder_dim, decoder_dim)  # attention network
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()


    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)


    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune


    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out,feature_map, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state

        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding = self.attention(feature_map[:batch_size_t],
                                                                h[:batch_size_t])
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds


        return predictions, encoded_captions, decode_lengths, sort_ind