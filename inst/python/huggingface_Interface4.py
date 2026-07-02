# Paths in python to task_finetune is set in R function
# from task_finetune import main as task_finetuner
# from run_mlm import main as mlm_finetuner
import json
import os, sys
import torch
import torch.nn.functional as F

import huggingface_hub
import transformers
from transformers import AutoConfig, AutoModel, AutoProcessor

try:
    from transformers.utils import logging
except ImportError:
    print("Warning: Unable to importing transformers.utils logging")

from platform import system


def set_logging_level(logging_level):
    """
    Set the logging level

    Parameters
    ----------
    logging_level : str
        set logging level, options: critical, error, warning, info, debug
    """
    logging_level = logging_level.lower()
    # default level is warning, which is in between "error" and "info"
    if logging_level in ['warn', 'warning']:
        logging.set_verbosity_warning()
    elif logging_level == "critical":
        logging.set_verbosity(50)
    elif logging_level == "error":
        logging.set_verbosity_error()
    elif logging_level == "info":
        logging.set_verbosity_info()
    elif logging_level == "debug":
        logging.set_verbosity_debug()
    else:
        print("Warning: Logging level {l} is not an option.".format(l=logging_level))
        print("\tUse one of: critical, error, warning, info, debug")


def set_hg_gated_access(access_token):
    """
    Local save of the access token for gated models on hg.
    
    Parameters
    ----------
    access_token : str
        Steps to get the access_token:
        1. Log in to your Hugging Face account.
        2. Click on your profile picture in the top right corner.
        3. Select ‘Settings’ from the dropdown menu.
        4. In the settings, you’ll find an option to generate a new token.
        Or, visit URL: https://huggingface.co/settings/tokens
    """
    huggingface_hub.login(access_token)
    print("Successfully login to Huggingface!")


def del_hg_gated_access():
    """
    Remove the access_token saved locally.

    """
    huggingface_hub.logout()
    print("Successfully logout to Huggingface!")


def get_device(device):
    """
    Get device and device number

    Parameters
    ----------
    device : str
        name of device: 'cpu', 'gpu', 'cuda', 'mps', or of the form 'gpu:k', 'cuda:k', or 'mps:0'
        where k is a specific device number

    Returns
    -------
    device : str
        final selected device name
    device_num : int
        device number, -1 for CPU
    """
    device = device.lower()
    if not device.startswith('cpu') and not device.startswith('gpu') and not device.startswith('cuda') and not device.startswith('mps'):
        print("device must be 'cpu', 'gpu', 'cuda', 'mps', or of the form 'gpu:k', 'cuda:k', or 'mps:0'")
        print("\twhere k is an integer value for the device")
        print("Trying CPUs")
        device = 'cpu'
    
    device_num = -1
    if device != 'cpu':
        attached = False
        
        if hasattr(torch.backends, "mps"):
            mps_available = torch.backends.mps.is_available()
        else:
            mps_available = False
        print(f"MPS for Mac available: {mps_available}")
        if torch.cuda.is_available():
            if device == 'gpu' or device == 'cuda': 
                # assign to first gpu device number
                device = 'cuda'
                device_num = list(range(torch.cuda.device_count()))[0]
                attached = True
            elif 'gpu:' in device or 'cuda:' in device:
                try:
                    device_num = int(device.split(":")[-1])
                    device = 'cuda:' + str(device_num)
                    attached = True
                except:
                    attached = False
                    print(f"Device number {str(device_num)} does not exist! Use 'device = gpus' to see available gpu numbers.")
            elif 'gpus' in device:
                device = 'cuda'
                device_num = list(range(torch.cuda.device_count()))
                device = [device + ':' + str(num1) for num1 in device_num]
                attached = True
                print(f"Running on {str(len(device))} GPUs!")
                print(f"Available gpus to set: \n {device}")
        elif "mps" in device:
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not built with MPS enabled.")
                else:
                    print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            else:
                device_num = 0 # list(range(torch.cuda.device_count()))[0]
                device = 'mps:' + str(device_num)
                attached = True
                print("Using Metal Performance Shaders (MPS) backend for GPU training acceleration!")
        else:
            attached = False
        if not attached:
            print("Unable to use MPS (Mac M1+), CUDA (GPU), using CPU")
            device = "cpu"
            device_num = -1

    return device, device_num


def set_tokenizer_parallelism(tokenizer_parallelism):
    if tokenizer_parallelism:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_audio_model(model, processor_only=False, config_only=False, hg_gated=False, hg_token="", trust_remote_code=False, for_transcription=False):
    """
    Get audio model and tokenizer from model string

    Parameters
    ----------
    model : str
        shortcut name for Hugging Face pretained model
        Full list https://huggingface.co/transformers/pretrained_models.html
    hg_gated : bool
        Set to True if the model is gated
    hg_token: str
        The token to access the gated model got in huggingface website
    
    Returns
    -------
    config
    processor
    model
    """
    if hg_gated:
        set_hg_gated_access(access_token=hg_token)
    else: 
        pass
    config = AutoConfig.from_pretrained(model)
    if not config_only:
        processor = AutoProcessor.from_pretrained(model)
        if for_transcription:
            from transformers import WhisperForConditionalGeneration
            transformer_model = WhisperForConditionalGeneration.from_pretrained(model, config=config, trust_remote_code=trust_remote_code)
        else:
            transformer_model = AutoModel.from_pretrained(model, config=config, trust_remote_code=trust_remote_code)
            
    if config_only:
        return config
    elif processor_only:
        return processor
    else:     
        return config, processor, transformer_model
    

def hgTransformerMLM(json_path, text_df_train, text_df_val, text_df_test, **kwargs):
    """
    Simple Python method for MLM fine tuning pretrained Hugging Face models
    
    Parameters
    ----------
    json_path : str
        Path to the json file containing the arguments for fine tuning model
    text_df_train : pandas dataframe
        Dataframe containing the text for training
    text_df_val : pandas dataframe
        Dataframe containing the text for validation
    text_df_test : pandas dataframe
        Dataframe containing the text for testing
        
    Returns
    -------
    None
    """
    args = json.load(open(json_path))
    return mlm_finetuner(args, text_df_train, text_df_val, text_df_test, **kwargs)


def hgTransformerFineTune(json_path, 
                            text_outcome_df_train, 
                            text_outcome_df_val, 
                            text_outcome_df_test,
                            pytorch_mps_high_watermark_ratio,
                            is_regression = True,
                            tokenizer_parallelism = False,
                            label_names = None, 
                            **kwargs):

    """
    Simple Python method for fine tuning pretrained Hugging Face models

    Parameters
    ----------
    json_path : str
        Path to the json file containing the arguments for fine tuning model
    text_outcome_df_train : pandas dataframe
        Dataframe containing the text and outcome variables for training
    text_outcome_df_val : pandas dataframe
        Dataframe containing the text and outcome variables for validation
    text_outcome_df_test : pandas dataframe
        Dataframe containing the text and outcome variables for testing
    is_regression : bool
        True if the outcome variable is continuous, False if the outcome variable is categorical
    label_names : list
        List of strings containing the class names for classification task
    
    Returns
    -------
    None
    """

    # Check if running on macOS and PYTORCH_MPS_HIGH_WATERMARK_RATIO is set to 'TRUE'
    if system() == 'Darwin' and pytorch_mps_high_watermark_ratio:
      os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
      print("Setting PYTORCH_MPS_HIGH_WATERMARK_RATIO to '0.0' on macOS with enabled flag.")


    args = json.load(open(json_path))
    return task_finetuner(args, text_outcome_df_train, text_outcome_df_val, text_outcome_df_test, is_regression, label_names, **kwargs)
     

# Transcription function uses Whisper
def hgTransformerTranscribe(
    audio_filepaths,
    model = 'whisper-tiny',
    device = 'cpu',
    tokenizer_parallelism = False,
    hg_gated = False,
    hg_token = "",
    trust_remote_code = False,
    logging_level = 'warning',
):
    """
    Simple Python method for embedding speech with pretained Hugging Face models

    Parameters
    ----------
    audio_filepaths : list
        list of audio filepaths, each is embedded separately
    model : str
        shortcut name for Hugging Face pretained model for speech-to-text task
        Full list https://huggingface.co/transformers/pretrained_models.html
    device : str
        name of device: 'cpu', 'gpu', or 'gpu:k' where k is a specific device number
    tokenizer_parallelism :  bool
        Whether to use device parallelization during tokenization
    hg_gated : bool
        Whether the accessed model is gated
    hg_token: str
        The token generated in huggingface website
    trust_remote_code : bool
        use a model with custom code on the Huggingface Hub
    logging_level : str
        set logging level, options: critical, error, warning, info, debug

    Returns
    -------
    all_transcripts : list, optional
        text
    """
    set_logging_level(logging_level)
    set_tokenizer_parallelism(tokenizer_parallelism)
    device, device_num = get_device(device)

    if not isinstance(audio_filepaths, list):
        audio_filepaths = [audio_filepaths]
    
    config, processor, transformer_model = get_audio_model(model, hg_gated=hg_gated, hg_token=hg_token, trust_remote_code=trust_remote_code, for_transcription=True)

    if device != 'cpu':
        transformer_model.to(device)
    transformer_model.eval()

    all_transcripts = []

    for audio_filepath in audio_filepaths:
        waveform = preprocess_audio(audio_filepath)
        audio_inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
        
        if device != 'cpu':
            audio_inputs = audio_inputs.to(device)

        try:
            with torch.no_grad():
                # Generate transcription
                generated_ids = transformer_model.generate(**audio_inputs)

            # Decode transcription
            transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            all_transcripts.append(transcript)
        except Exception as e:
            print(f'\"{audio_filepath}\" failed with the following error:')
            print(Warning(e))
    
    if hg_gated:
        del_hg_gated_access()
    return all_transcripts


def hgTransformerGetEmbedding(
    audio_filepaths,
    audio_transcriptions = None,
    model = 'whisper-tiny',
    use_decoder = False,
    tokenizer_parallelism = False,
    model_max_length = None,
    device = 'cpu',
    hg_gated = False,
    hg_token = "",
    trust_remote_code = False,
    logging_level = 'warning',
):
    """
    Simple Python method for embedding speech with pretained Hugging Face models

    Parameters
    ----------
    audio_filepaths : list
        list of audio filepaths, each is embedded separately
    audio_transcriptions : list
        (optional) list of audio transcriptions, to be used for Whisper's decoder-based embeddings
    model : str
        shortcut name for Hugging Face pretained model
        Full list https://huggingface.co/transformers/pretrained_models.html
    use_decoder : bool
        Whether to use Whisper's decoder last hidden state representation
        (Note: audio_transcriptions must be provided if this option is set to true)
    tokenizer_parallelism :  bool
        Whether to use device parallelization during tokenization
    model_max_length : int
        maximum length of the tokenized text
    device : str
        name of device: 'cpu', 'gpu', or 'gpu:k' where k is a specific device number
    hg_gated : bool
        Whether the accessed model is gated
    hg_token: str
        The token generated in huggingface website
    trust_remote_code : bool
        use a model with custom code on the Huggingface Hub
    logging_level : str
        set logging level, options: critical, error, warning, info, debug

    Returns
    -------
    all_embs : list
        embeddings for each item in text_strings
    """
    set_logging_level(logging_level)
    set_tokenizer_parallelism(tokenizer_parallelism)
    device, device_num = get_device(device)

    # check and adjust input types
    if use_decoder:
        if audio_transcriptions is None:
            raise AssertionError('audio_transcriptions must be provided if use_decoder is True')
        else:
            if not isinstance(audio_transcriptions, list):
                audio_transcriptions = [audio_transcriptions]
    
    if not isinstance(audio_filepaths, list):
        audio_filepaths = [audio_filepaths]

    config, processor, transformer_model = get_audio_model(model, hg_gated=hg_gated, hg_token=hg_token, trust_remote_code=trust_remote_code)

    if device != 'cpu':
        transformer_model.to(device)
    transformer_model.eval()

    all_embs = []

    for i, audio_filepath in enumerate(audio_filepaths):
        waveform = preprocess_audio(audio_filepath)
        audio_inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
        
        if device != 'cpu':
            audio_inputs = audio_inputs.to(device)

        try:
            with torch.no_grad(): 
                # Wav2Vec Embedding Generation
                if isinstance(transformer_model, transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model):
                    embedding = transformer_model(**audio_inputs).last_hidden_state.mean(1).squeeze()

                # Whisper Embedding Generation (Encoder Representation)
                elif isinstance(transformer_model, transformers.models.whisper.modeling_whisper.WhisperModel):
                    if use_decoder:
                        # Whisper-based tokenization
                        tokens = processor.tokenizer(
                            audio_transcriptions[i],
                            # padding=True,
                            # truncation=True,
                            max_length=model_max_length,
                            return_tensors='pt'
                        ).to(device)

                        # Get WhiSPA's MEAN/LAST token
                        whis_embs = model(
                            audio_inputs['input_values'],
                            tokens['input_ids'],
                            tokens['attention_mask']
                        )
                    else:
                        embedding = transformer_model.encoder(**audio_inputs).last_hidden_state.mean(1).squeeze()
                
                else:
                    raise AssertionError('Not implemented yet...')

#            all_embs.append(embedding)
            all_embs.append(embedding.cpu().numpy().tolist())
        except Exception as e:
            print(f'\"{audio_filepath}\" failed with the following error:')
            print(Warning(e))

    if hg_gated:
        del_hg_gated_access()
    return all_embs


def preprocess_audio(audio_path):
    import soundfile as sf
    import librosa
    # Load with soundfile/librosa rather than torchaudio.load, which routes
    # through torchcodec and can fail to load its shared library in some
    # environments (e.g. the diarisation env). soundfile reads the same files.
    _sf_data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    # Convert stereo (or multi-channel) to mono if needed   
    if _sf_data.shape[1] > 1:
        _sf_data = _sf_data.mean(axis=1)
    else:
        _sf_data = _sf_data[:, 0]
    # Resample if necessary (Whisper requires 16kHz input)
    if sample_rate != 16000:
        _sf_data = librosa.resample(_sf_data, orig_sr=sample_rate, target_sr=16000)
    # Return shape (1, samples) to match the previous torchaudio.load output
    return torch.from_numpy(_sf_data).unsqueeze(0)


# def mean_pooling(embeddings, attention_mask):
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
#     return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


## MAIN FOR TESTING PURPOSES
#if __name__ == '__main__':
#    embs = hgTransformerGetEmbedding(
#        audio_filepaths = '',
#        model = 'openai/whisper-tiny', # facebook/wav2vec2-base-960h
#        use_decoder = False,
#        tokenizer_parallelism = False,
#        model_max_length = None,
#        device = 'cpu',
#        hg_gated = False,
#        hg_token = "",
#        trust_remote_code = False,
#        logging_level = 'warning',
#    )
#
#    print(embs[0].shape)
#
#    transcripts = hgTransformerTranscribe(
#        audio_filepaths = '',
#        model = 'openai/whisper-tiny', # facebook/wav2vec2-base-960h
#        tokenizer_parallelism = False,
#        device = 'cpu',
#        hg_gated = False,
#        hg_token = "",
#        trust_remote_code = False,
#        logging_level = 'warning',
#    )
#
#    print(transcripts[0])
