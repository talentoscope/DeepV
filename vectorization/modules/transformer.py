import torch
import torch.nn as nn

from vectorization.modules.base import ParameterizedModule

from ._transformer_modules import TransformerLayer, get_sinusoid_encoding_table


class _InternalSequentialTransformerDecoder(nn.Module):
    def __init__(self, feature_dim=128, ffn_dim=512, n_head=8, num_layers=1, **kwargs):
        """
        :param feature_dim: Wq, Wk, Wv embedding matrixes share this dimension
        :param ffn_dim: size of FC layers in TransformerLayers
        :param n_head: number of heads in TransformerLayers
        :param num_layers: number of TransformerLayers stacked together
        """
        super(_InternalSequentialTransformerDecoder, self).__init__()
        self.transformer = nn.Sequential(
            *(TransformerLayer(feature_dim, d_inner=ffn_dim, n_head=n_head) for _ in range(num_layers))
        )
        self.feature_dim = feature_dim

    def forward(self, conv_features, hidden_encoding):
        # conv_features: [b, h * w, c]
        # hidden_encoding: [b, n, c]
        h_dec = hidden_encoding

        for layer in self.transformer:
            h_dec = layer(h_dec, conv_features)

        return h_dec


class TransformerBase(ParameterizedModule):
    def __init__(self, feature_dim=128, ffn_dim=512, n_head=8, num_layers=1, **kwargs):
        """
        :param feature_dim: Wq, Wk, Wv embedding matrixes share this dimension
        :param ffn_dim: size of FC layers in TransformerLayers
        :param n_head: number of heads in TransformerLayers
        :param num_layers: number of TransformerLayers stacked together
        """
        super(TransformerBase, self).__init__(**kwargs)
        self.decoder = _InternalSequentialTransformerDecoder(
            feature_dim=feature_dim,
            ffn_dim=ffn_dim,
            n_head=n_head,
            num_layers=num_layers,
        )

        self.feature_dim = feature_dim


class TransformerDecoder(TransformerBase):
    def forward(self, conv_features, max_lines):
        """
        :param conv_features: [b, c, h, w] batch of image conv features
        :param max_lines: how many lines per image to predict
        """
        sine_enc = get_sinusoid_encoding_table(max_lines, self.feature_dim, scale=1)[None]
        h_dec = torch.cat([sine_enc] * conv_features.shape[0], dim=0)  # [b, max_lines, feature_dim]
        h_dec = h_dec.to(conv_features.device)
        decoding = self.decoder(conv_features, h_dec)
        return decoding


class TransformerDiscriminator(TransformerBase):
    LINE_DIM = 6

    def __init__(self, **kwargs):
        super(TransformerDiscriminator, self).__init__(**kwargs)
        self.fc = nn.Linear(self.LINE_DIM, self.feature_dim)

    def forward(self, conv_features, predicted_lines):
        """
        :param conv_features: [b, c, h, w] batch of image conv features
        :param predicted_lines: [b, n, line_dim] batch of predicted n lines per image
        """
        h_dec = self.fc(predicted_lines)
        decoding = self.decoder(conv_features, h_dec)
        return decoding


class VariableLengthTransformerDecoder(TransformerBase):
    def __init__(self, feature_dim=128, ffn_dim=512, n_head=8, num_layers=1, max_primitives=20, **kwargs):
        """
        :param feature_dim: Wq, Wk, Wv embedding matrixes share this dimension
        :param ffn_dim: size of FC layers in TransformerLayers
        :param n_head: number of heads in TransformerLayers
        :param num_layers: number of TransformerLayers stacked together
        :param max_primitives: maximum number of primitives to generate
        """
        super(VariableLengthTransformerDecoder, self).__init__(feature_dim=feature_dim, ffn_dim=ffn_dim, n_head=n_head, num_layers=num_layers, **kwargs)
        self.max_primitives = max_primitives
        # Output layer that predicts primitive parameters
        self.output_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, conv_features):
        """
        :param conv_features: [b, seq_len, c] batch of image conv features
        :return: [b, max_primitives, feature_dim] predictions for each position
        """
        batch_size = conv_features.shape[0]

        # Initialize with learned start token (could be zeros or learned embedding)
        start_embedding = torch.zeros(batch_size, 1, self.feature_dim, device=conv_features.device)
        # Or use a learned start token:
        # self.start_token = nn.Parameter(torch.randn(1, 1, self.feature_dim))
        # start_embedding = self.start_token.expand(batch_size, -1, -1)

        # Generate positional encodings for all positions
        pos_encodings = get_sinusoid_encoding_table(self.max_primitives, self.feature_dim, scale=1)[None].to(conv_features.device)
        pos_encodings = pos_encodings.expand(batch_size, -1, -1)  # [b, max_primitives, feature_dim]

        # Initialize sequence with start embedding
        current_sequence = start_embedding

        outputs = []

        for i in range(self.max_primitives):
            # Get current positional encoding
            current_pos = pos_encodings[:, i:i+1]  # [b, 1, feature_dim]

            # Add positional encoding to current input
            current_input = current_sequence + current_pos

            # Decode current step
            decoded = self.decoder(conv_features, current_input)

            # Project to output space
            step_output = self.output_proj(decoded.squeeze(1))  # [b, feature_dim]

            outputs.append(step_output.unsqueeze(1))

            # Use this output as input for next step
            current_sequence = step_output.unsqueeze(1)

        return torch.cat(outputs, dim=1)


class NonAutoregressiveTransformerDecoder(TransformerBase):
    def __init__(self, feature_dim=128, ffn_dim=512, n_head=8, num_layers=1, max_primitives=20, **kwargs):
        """
        Non-autoregressive transformer decoder that predicts all primitives simultaneously.
        
        :param feature_dim: Wq, Wk, Wv embedding matrixes share this dimension
        :param ffn_dim: size of FC layers in TransformerLayers
        :param n_head: number of heads in TransformerLayers
        :param num_layers: number of TransformerLayers stacked together
        :param max_primitives: maximum number of primitives to generate
        """
        super(NonAutoregressiveTransformerDecoder, self).__init__(
            feature_dim=feature_dim, ffn_dim=ffn_dim, n_head=n_head, num_layers=num_layers, **kwargs
        )
        self.max_primitives = max_primitives
        
        # Output layer that predicts primitive parameters
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        # Count predictor to estimate number of primitives
        self.count_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, max_primitives + 1)  # +1 for 0 to max_primitives
        )

    def forward(self, conv_features):
        """
        :param conv_features: [b, seq_len, c] batch of image conv features
        :return: predictions: [b, max_primitives, feature_dim], 
                 counts: [b, max_primitives + 1] probability distribution over count
        """
        batch_size = conv_features.shape[0]

        # Generate positional encodings for all positions simultaneously
        pos_encodings = get_sinusoid_encoding_table(self.max_primitives, self.feature_dim, scale=1)[None].to(conv_features.device)
        pos_encodings = pos_encodings.expand(batch_size, -1, -1)  # [b, max_primitives, feature_dim]

        # Initialize all positions with positional encodings (no autoregressive dependency)
        h_dec = pos_encodings

        # Apply transformer layers in parallel
        decoding = self.decoder(conv_features, h_dec)  # [b, max_primitives, feature_dim]

        # Project to output space
        predictions = self.output_proj(decoding)  # [b, max_primitives, feature_dim]

        # Predict primitive count using global features (average pooling)
        global_features = torch.mean(conv_features, dim=1)  # [b, c]
        if global_features.shape[-1] != self.feature_dim:
            # If conv features have different dimension, add a projection
            global_proj = nn.Linear(conv_features.shape[-1], self.feature_dim).to(conv_features.device)
            global_features = global_proj(global_features)
        
        counts = self.count_predictor(global_features)  # [b, max_primitives + 1]

        return predictions, counts


transformer_module_by_kind = {
    "transformer_decoder": TransformerDecoder,
    "variable_transformer_decoder": VariableLengthTransformerDecoder,
    "non_autoregressive_transformer_decoder": NonAutoregressiveTransformerDecoder,
    "transformer_discriminator": TransformerDiscriminator,
}
