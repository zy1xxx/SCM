import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.attention_processor import AttnProcessor
from einops import rearrange, repeat
class AttnStore():
    def __init__(self):
        super().__init__()
        self.attn = None
        self.subject_index=None
        self.story_length=None
        self.threshold=100

    def set_story_length(self,story_length):
        self.story_length=story_length
    def set_attention_probs(self, attn):
        self.attn = attn
    def set_subject_index(self,subject_index):
        self.subject_index=subject_index
    def get_mask(self):
        if self.attn is None:
            raise ValueError("attn map not set")
        elif self.subject_index is None and self.story_length is None:
            raise ValueError("subject index or story length not set")
        else:
            ##construct character mask
            attn=rearrange(self.attn,"(b f) s d -> f b s d",f=self.story_length)
            character_mask_ls=[]
            for i in range(self.story_length):
                index=self.subject_index[i]
                f_attn=attn[i]
                if index==[]:
                    character_mask=torch.zeros(f_attn[:,:,0].shape).to(f_attn.device).to(f_attn.dtype)
                else:
                    f_attn=torch.mean(f_attn[:,:,index],dim=2)
                    f_attn = 255 * f_attn / torch.max(f_attn,dim=1,keepdim=True)[0] #normalize
                    character_mask_zero=torch.zeros(f_attn.shape).to(f_attn.device).to(f_attn.dtype)
                    character_mask=character_mask_zero.masked_fill(f_attn>self.threshold,1)
                character_mask_ls.append(character_mask)
            character_mask=torch.stack(character_mask_ls,dim=0)
            character_mask=rearrange(character_mask,"f b s->(b s) f")

            ##construct attention mask
            original_vector=character_mask.unsqueeze(-2)
            column_vector = torch.transpose(original_vector, -1, -2)
            mask = column_vector * original_vector

            #Fill in 0 on the diagonal to ensure that it is not masked.
            identity_matrix = torch.eye(mask.shape[-1],dtype=bool)
            diagonal_matrices = identity_matrix.unsqueeze(0).expand(mask.shape[0], -1, -1).to(mask.device)
            with_self_mask=mask.masked_fill(diagonal_matrices, 1)

            attn_mask=torch.zeros(with_self_mask.shape).to(with_self_mask.device).to(with_self_mask.dtype)
            attn_mask.masked_fill_(with_self_mask==0,float('-inf'))
            attn_mask.masked_fill_(attn_mask==1,0)
            return attn_mask

class CrossAttnProcessor:
    def __init__(self, attnstore:AttnStore):
        super().__init__()
        self.attnstore = attnstore
    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length,batch_size)

        query = attn.to_q(hidden_states)
        
        is_cross = encoder_hidden_states is not None
        
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.attnstore.set_attention_probs(attention_probs)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class TempAttnProcessor:
    def __init__(self, attnstore:AttnStore):
        super().__init__()
        self.attnstore = attnstore

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length,batch_size)

        query = attn.to_q(hidden_states)
        
        is_cross = encoder_hidden_states is not None
        
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_mask=self.attnstore.get_mask()

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def register_attention_control(unet, attnstore: AttnStore):
    attn_procs = {}
    for name in unet.attn_processors.keys():    
        if name.startswith("mid_block") or name.startswith("up_blocks") or name.startswith("down_blocks"):    
            if name.endswith("attn2.processor"):
                attn_procs[name] = CrossAttnProcessor(
                    attnstore=attnstore
                )
            elif name.endswith("temp.processor"):
                attn_procs[name] = TempAttnProcessor(
                    attnstore=attnstore
                )
            else:
                attn_procs[name] = AttnProcessor()
        else:
            continue

    unet.set_attn_processor(attn_procs)   #set attn_processor


