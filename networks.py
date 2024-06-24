import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import models.TCN
from config import args


class Header(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, channels_list: list = []) -> None:
        super(Header, self).__init__()
        self.channels_list = [input_channels] + channels_list + [output_channels]
        self.layers = nn.ModuleList()
        for i in range(len(self.channels_list) - 1):
            self.layers.append(nn.Linear(self.channels_list[i], self.channels_list[i + 1]))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = F.softmax(x, dim=1)
        return x
        
class FeatureEmbed(nn.Module):
    def __init__(self, input_size: int, output_size: int, droupout: float) -> None:
        super(FeatureEmbed, self).__init__()
        self.fc1 = nn.Linear(input_size, args.feature_encoder_hidden_size)
        self.bn = nn.BatchNorm1d(args.feature_encoder_hidden_size)
        self.dropout = nn.Dropout(droupout)
        self.fc2 = nn.Linear(args.feature_encoder_hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        


class triFNRIS(nn.Module):
    def __init__(self,
                 encoder_name: str, decoder_name: str, header_name: str,
                 input_channels: int, output_channels: int, embedding_dim: int, dropout: float,
                 hap_embedding_dim: int = None, sad_embedding_dim: int = None, rs_embedding_dim: int = None,
                 TCN_encoder_list_channels: list = None, TCN_hidden_size: int = None, TCN_kernel_size: int = None,
                 header_channels_list: list = None, is_encoder_share_weights: bool = True) -> None:
        super(triFNRIS, self).__init__()
        # ---------- Encoder, Decoder, and Header ----------
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.header_name = header_name
        # ---------- Model Settings ----------
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.is_encoder_share_weights = is_encoder_share_weights
        # ---------- TCN Encoder ----------
        self.TCN_encoder_list_channels = TCN_encoder_list_channels if TCN_encoder_list_channels else args.TCN_list_channels
        self.TCN_hidden_size = TCN_hidden_size if TCN_hidden_size else args.TCN_hidden_size
        self.TCN_kernel_size = TCN_kernel_size if TCN_kernel_size else args.TCN_kernel_size
        # ---------- 2_class_MLP Header ----------
        self.header_channels_list = header_channels_list if header_channels_list else args.header_channels_list
        # ---------- Embeddings ----------
        self.hap_embedding_dim = hap_embedding_dim if hap_embedding_dim else embedding_dim
        self.sad_embedding_dim = sad_embedding_dim if sad_embedding_dim else embedding_dim
        self.rs_embedding_dim = rs_embedding_dim if rs_embedding_dim else embedding_dim
        # ---------- Init ----------
        self.__init_encoders__()
        self.__init_decoders__()
        self.__init_headers__()
        self.__init_embeddings__()
        self.sequence_weights = nn.Parameter(torch.empty(3).normal_(mean=0, std=0.02), requires_grad=True)
        if args.emotional_signals in ["hap-sad", "hap-rs", "sad-rs"]:
            num_embeddings = 4
        elif args.emotional_signals in ["hap", "sad", "rs"]:
            num_embeddings = 2
        else:
            num_embeddings = 6
        num_embeddings = num_embeddings / 2 if (args.is_sequence_only or args.is_features_only) else num_embeddings
        self.embedding_weights = nn.Parameter(torch.empty(num_embeddings).normal_(mean=0, std=0.02), requires_grad=True)
    
    def __init_encoders__(self):
        if self.encoder_name == "TCN":
            self.init_encoders_TCN()
        else:
            raise RuntimeError(f"no encoder names {self.encoder_name}")
        self.features_encoder = FeatureEmbed(self.input_channels * args.num_extracted_features, self.embedding_dim, self.dropout)
        
    
    def __init_decoders__(self):
        if self.decoder_name == "MLP":
            self.init_decoders_MLP()
        else:
            raise RuntimeError(f"no decoder names {self.decoder_name}")
    
    def __init_headers__(self):
        if self.header_name == "2_classes_MLP":
            self.init_headers_2_class_MLP()
        else:
            raise RuntimeError(f"no header names {self.header_name}")
    
    def __init_embeddings__(self):
        self.hap_embedding = nn.Parameter(torch.empty(1, self.hap_embedding_dim).normal_(mean=0, std=0.02), requires_grad=True)
        self.sad_embedding = nn.Parameter(torch.empty(1, self.sad_embedding_dim).normal_(mean=0, std=0.02), requires_grad=True)
        self.rs_embedding = nn.Parameter(torch.empty(1, self.rs_embedding_dim).normal_(mean=0, std=0.02), requires_grad=True)
        self.hap_feat_embedding = nn.Parameter(torch.empty(1, self.hap_embedding_dim).normal_(mean=0, std=0.02), requires_grad=True)
        self.sad_feat_embedding = nn.Parameter(torch.empty(1, self.sad_embedding_dim).normal_(mean=0, std=0.02), requires_grad=True)
        self.rs_feat_embedding = nn.Parameter(torch.empty(1, self.rs_embedding_dim).normal_(mean=0, std=0.02), requires_grad=True)
    
    def init_encoders_TCN(self):
        if self.is_encoder_share_weights:
            encoder = models.TCN.TCN(self.input_channels, self.embedding_dim, self.TCN_encoder_list_channels, 
                                     self.TCN_hidden_size, self.TCN_kernel_size, self.dropout)
            self.hap_encoder = self.sad_encoder = self.rs_encoder = encoder
        else:
            self.hap_encoder = models.TCN.TCN(self.input_channels, self.embedding_dim, self.TCN_encoder_list_channels, 
                                            self.TCN_hidden_size, self.TCN_kernel_size, self.dropout)
            self.sad_encoder = models.TCN.TCN(self.input_channels, self.embedding_dim, self.TCN_encoder_list_channels, 
                                            self.TCN_hidden_size, self.TCN_kernel_size, self.dropout)
            self.rs_encoder = models.TCN.TCN(self.input_channels, self.embedding_dim, self.TCN_encoder_list_channels, 
                                            self.TCN_hidden_size, self.TCN_kernel_size, self.dropout)
    
    def init_decoders_MLP(self):
        self.decoder = nn.Linear((self.hap_embedding_dim + self.sad_embedding_dim + self.rs_embedding_dim), self.embedding_dim)
        
    def init_headers_2_class_MLP(self):
        self.header = Header(self.embedding_dim, self.output_channels, )
        
    def combine(self, x):
        num_channels = int(x.shape[1] / 3)
        weights = F.softmax(self.sequence_weights, dim=0)
        HbO = weights[0] * x[:, 0: num_channels, :]
        HbR = weights[1] * x[:, num_channels: 2 * num_channels, :]
        HbT = weights[2] * x[:, 2 * num_channels: 3 * num_channels, :]
        x = HbO + HbR + HbT
        return x
    
    def forward(self, xs):
        if args.is_use_extracted_features:
            hap, sad, rs, hap_feat, sad_feat, rs_feat = xs
            if args.is_combine:
                hap = self.combine(hap)
                sad = self.combine(sad)
                rs = self.combine(rs)
                hap_feat = self.combine(hap_feat).view(hap_feat.shape[0], -1)
                sad_feat = self.combine(sad_feat).view(sad_feat.shape[0], -1)
                rs_feat = self.combine(rs_feat).view(rs_feat.shape[0], -1)
            else:
                print(rs_feat.shape)
                hap_feat = hap_feat.view(hap_feat.shape[0], -1)
                sad_feat = sad_feat.view(hap_feat.shape[0], -1)
                rs_feat = rs_feat.view(hap_feat.shape[0], -1)
                print(rs_feat.shape)
            
            if args.is_use_type_emb:
                hap_embedding = self.hap_encoder(hap) + self.hap_embedding
                sad_embedding = self.sad_encoder(sad) + self.sad_embedding
                rs_embedding = self.rs_encoder(rs) + self.rs_embedding
                hap_feat_embedding = self.features_encoder(hap_feat) + self.hap_feat_embedding
                sad_feat_embedding = self.features_encoder(sad_feat) + self.sad_feat_embedding
                rs_feat_embedding = self.features_encoder(rs_feat) + self.rs_feat_embedding
            else:
                hap_embedding = self.hap_encoder(hap)
                sad_embedding = self.sad_encoder(sad)
                rs_embedding = self.rs_encoder(rs)
                hap_feat_embedding = self.features_encoder(hap_feat)
                sad_feat_embedding = self.features_encoder(sad_feat)
                rs_feat_embedding = self.features_encoder(rs_feat)
            
            weights = F.softmax(self.embedding_weights, dim=0)
            if args.is_sequence_only:
                embedding = weights[0] * hap_embedding + weights[1] * sad_embedding + weights[2] * rs_embedding
            if args.is_features_only:
                embedding = weights[0] * hap_feat_embedding + weights[1] * sad_feat_embedding + weights[2] * rs_feat_embedding
            if not (args.is_sequence_only or args.is_features_only):
                if args.emotional_signals == "hap-sad-rs":
                    embedding = weights[0] * hap_embedding + weights[1] * sad_embedding + weights[2] * rs_embedding + weights[3] * hap_feat_embedding + weights[4] * sad_feat_embedding + weights[5] * rs_feat_embedding
                elif args.emotional_signals == "hap-sad":
                    embedding = weights[0] * hap_embedding + weights[1] * sad_embedding + weights[2] * hap_feat_embedding + weights[3] * sad_feat_embedding
                elif args.emotional_signals == "hap-rs":
                    embedding = weights[0] * hap_embedding + weights[1] * rs_embedding + weights[2] * hap_feat_embedding + weights[3] * rs_feat_embedding
                elif args.emotional_signals == "sad-rs":
                    embedding = weights[0] * sad_embedding + weights[1] * rs_embedding + weights[2] * sad_feat_embedding + weights[3] * rs_feat_embedding
                elif args.emotional_signals == "hap":
                    embedding = weights[0] * hap_embedding + weights[1] * hap_feat_embedding
                elif args.emotional_signals == "sad":
                    embedding = weights[0] * sad_embedding + weights[1] * sad_feat_embedding
                elif args.emotional_signals == "rs":
                    embedding = weights[0] * rs_embedding + weights[1] * rs_feat_embedding
                else:
                    raise RuntimeError(f"no emotional signals name {args.emotional_signals}")
            y = self.header(embedding)
            return y
        else:
            hap, sad, rs = xs
            hap_embedding = self.hap_encoder(hap) + self.hap_embedding
            sad_embedding = self.sad_encoder(sad) + self.sad_embedding
            rs_embedding = self.rs_encoder(rs) + self.rs_embedding
            fused_embedding = torch.concat((hap_embedding, sad_embedding, rs_embedding), dim=1)
            fused_embedding = self.decoder(fused_embedding)
            y = self.header(fused_embedding)
            return y
    