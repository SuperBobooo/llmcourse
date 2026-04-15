import torch
import torch.nn as nn

from models.decoder import Decoder
from models.embeddings import TokenEmbedding
from models.encoder import Encoder
from utils.masks import make_cross_attention_mask, make_decoder_self_attention_mask


class TransformerTranslationModel(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_id: int,
        tgt_pad_id: int,
        config,
    ) -> None:
        super().__init__()
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.d_model = config.d_model

        self.src_embedding = TokenEmbedding(
            vocab_size=src_vocab_size,
            d_model=config.d_model,
            pad_id=src_pad_id,
            dropout=config.dropout,
        )
        self.tgt_embedding = TokenEmbedding(
            vocab_size=tgt_vocab_size,
            d_model=config.d_model,
            pad_id=tgt_pad_id,
            dropout=config.dropout,
        )

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.output_projection = nn.Linear(config.d_model, tgt_vocab_size, bias=False)

        self.apply(self._reset_parameters)

    def _reset_parameters(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.d_model ** -0.5)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def encode(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_embeddings = self.src_embedding(src_ids)
        return self.encoder(src_embeddings, src_mask=src_mask)

    def decode(
        self,
        tgt_input_ids: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: torch.Tensor | None,
        cross_attn_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tgt_embeddings = self.tgt_embedding(tgt_input_ids)
        return self.decoder(
            tgt_embeddings,
            memory=memory,
            self_attn_mask=self_attn_mask,
            cross_attn_mask=cross_attn_mask,
        )

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        src_mask: torch.Tensor | None,
        tgt_self_attn_mask: torch.Tensor | None,
        cross_attn_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory, encoder_aux_loss = self.encode(src_ids, src_mask=src_mask)
        decoder_hidden, decoder_aux_loss = self.decode(
            tgt_input_ids=tgt_input_ids,
            memory=memory,
            self_attn_mask=tgt_self_attn_mask,
            cross_attn_mask=cross_attn_mask,
        )
        logits = self.output_projection(decoder_hidden)
        return logits, encoder_aux_loss + decoder_aux_loss

    @torch.no_grad()
    def greedy_decode(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
    ) -> torch.Tensor:
        self.eval()
        batch_size = src_ids.size(0)
        device = src_ids.device

        memory, _ = self.encode(src_ids, src_mask=src_mask)
        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_self_attn_mask = make_decoder_self_attention_mask(
                generated, pad_id=self.tgt_pad_id
            )
            cross_attn_mask = make_cross_attention_mask(
                generated, src_ids, pad_id=self.src_pad_id
            )

            decoder_hidden, _ = self.decode(
                tgt_input_ids=generated,
                memory=memory,
                self_attn_mask=tgt_self_attn_mask,
                cross_attn_mask=cross_attn_mask,
            )
            next_token_logits = self.output_projection(decoder_hidden[:, -1:, :])
            next_token = next_token_logits.argmax(dim=-1)

            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, eos_id),
                next_token,
            )
            generated = torch.cat([generated, next_token], dim=1)
            finished |= next_token.squeeze(-1).eq(eos_id)

            if finished.all():
                break

        return generated
