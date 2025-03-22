import numpy as np
from typing import Tuple
import re

class TranscriptionEvaluator:
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess text by removing extra whitespace and converting to lowercase
        """
        # Remove multiple spaces and convert to lowercase
        text = re.sub(r'\s+', ' ', text.lower().strip())
        # Remove punctuation except apostrophes
        text = re.sub(r'[^\w\s\']', '', text)
        return text

    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate (WER) between reference and hypothesis texts
        WER = (S + D + I) / N
        where:
        S = number of substitutions
        D = number of deletions
        I = number of insertions
        N = number of words in reference
        """
        # Preprocess texts
        reference = TranscriptionEvaluator.preprocess_text(reference)
        hypothesis = TranscriptionEvaluator.preprocess_text(hypothesis)
        
        # Convert to word arrays
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        # Create distance matrix
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
        
        # Initialize first row and column
        for i in range(len(ref_words) + 1):
            d[i, 0] = i
        for j in range(len(hyp_words) + 1):
            d[0, j] = j
            
        # Compute minimum edit distance
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i, j] = d[i-1, j-1]
                else:
                    substitution = d[i-1, j-1] + 1
                    insertion = d[i, j-1] + 1
                    deletion = d[i-1, j] + 1
                    d[i, j] = min(substitution, insertion, deletion)
                    
        # Calculate WER
        wer = float(d[len(ref_words), len(hyp_words)]) / len(ref_words)
        return wer

    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate (CER) between reference and hypothesis texts
        CER = (S + D + I) / N
        where:
        S = number of character substitutions
        D = number of character deletions
        I = number of character insertions
        N = number of characters in reference
        """
        # Preprocess texts
        reference = TranscriptionEvaluator.preprocess_text(reference)
        hypothesis = TranscriptionEvaluator.preprocess_text(hypothesis)
        
        # Create distance matrix
        d = np.zeros((len(reference) + 1, len(hypothesis) + 1))
        
        # Initialize first row and column
        for i in range(len(reference) + 1):
            d[i, 0] = i
        for j in range(len(hypothesis) + 1):
            d[0, j] = j
            
        # Compute minimum edit distance
        for i in range(1, len(reference) + 1):
            for j in range(1, len(hypothesis) + 1):
                if reference[i-1] == hypothesis[j-1]:
                    d[i, j] = d[i-1, j-1]
                else:
                    substitution = d[i-1, j-1] + 1
                    insertion = d[i, j-1] + 1
                    deletion = d[i-1, j] + 1
                    d[i, j] = min(substitution, insertion, deletion)
                    
        # Calculate CER
        cer = float(d[len(reference), len(hypothesis)]) / len(reference)
        return cer

    @staticmethod
    def evaluate(reference: str, hypothesis: str) -> Tuple[float, float]:
        """
        Evaluate transcription quality using both WER and CER
        """
        wer = TranscriptionEvaluator.calculate_wer(reference, hypothesis)
        cer = TranscriptionEvaluator.calculate_cer(reference, hypothesis)
        return wer, cer

def main():
    # Ground truth transcription
    ground_truth = """Hello? This is the book. It will be a bank of 1302 with international speedway pull apart. Okay. We're being robbed. Okay. Where are you at in the bank? He's saying he has a bomb. He's saying he has a tager. Okay. It's white male, black male. White male, black male, black male. White male. Okay. Black beanie, black hat. Light beige. What? Okay. Okay, Wells, can you tell me about him? Where's he at in the bank? He's very scared. He's saying he's got a bomb. Where I'm in the kitchen on that lunch. He came through with a bag."""

    # Dummy transcription with approximately 15 errors
    dummy_transcription = """Hello? This is the book. It will be a bank at 1302 with international speedway pull apart. Ok. Were being robed. Okay. Where are you at in the bank? He's saying he has a bom. He's saying he has a taser. Okay. Its white male, black male. White male, black male, black male. White male. Okay. Black benie, black hat. Light beige. What? Okay. Okay, Wells, can you tell me about him? Where is he at in the bank? He is very scared. He's saying he's got a bomb. Where I'm in the kichen on that lunch. He came through with a bag."""

    # Create evaluator instance
    evaluator = TranscriptionEvaluator()
    
    # Calculate WER and CER
    wer, cer = evaluator.evaluate(ground_truth, dummy_transcription)
    
    # Print results
    print(f"Word Error Rate (WER): {wer:.4f}")
    print(f"Character Error Rate (CER): {cer:.4f}")
    
    # Print detailed analysis
    print("\nDetailed Analysis:")
    print("-" * 50)
    print(f"Total words in reference: {len(ground_truth.split())}")
    print(f"Total characters in reference: {len(ground_truth)}")
    print(f"Approximate number of word errors: {int(wer * len(ground_truth.split()))}")
    print(f"Approximate number of character errors: {int(cer * len(ground_truth))}")

if __name__ == "__main__":
    main()
