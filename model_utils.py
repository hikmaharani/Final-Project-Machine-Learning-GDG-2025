import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os

class NERModel:
    def __init__(self, model_path="./models/ner_models"):
        """
        Initialize NER model
        
        Args:
            model_path: Path ke folder model
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔄 Loading model from {model_path}...")
        print(f"🖥️  Using device: {self.device}")
        
        # Check if model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Load tokenizer & model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Get label mapping
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id
            
            print(f"✅ Model loaded successfully!")
            print(f"📊 Total labels: {len(self.id2label)}")
            print(f"🏷️  Labels: {list(self.label2id.keys())}")
            
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def predict(self, text):
        """
        Predict NER entities from text
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of entities with labels
            Example: [
                {"text": "Joko Widodo", "label": "PERSON", "start": 0, "end": 11, "confidence": 0.99},
                {"text": "Indonesia", "label": "PLACE", "start": 32, "end": 41, "confidence": 0.95}
            ]
        """
        # Tokenize with offset mapping
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0]
            # Get confidence scores (probabilities)
            probs = torch.softmax(outputs.logits, dim=2)[0]
        
        # Convert to labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_labels = [self.id2label[pred.item()] for pred in predictions]
        confidences = [probs[i][pred.item()].item() for i, pred in enumerate(predictions)]
        
        # Extract entities with positions
        entities = []
        current_entity = None
        
        for idx, (token, label, offset, confidence) in enumerate(zip(tokens, predicted_labels, offset_mapping, confidences)):
            # Skip special tokens
            if token in ['<s>', '</s>', '<pad>', '<unk>']:  # XLM-RoBERTa special tokens
                continue
            
            start, end = offset.tolist()
            
            # Skip if offset is (0, 0) - means it's a special token
            if start == 0 and end == 0 and idx != 0:
                continue
            
            # Handle BIO tagging
            if label.startswith("B-"):
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)
                
                # Start new entity
                entity_type = label[2:]  # Remove "B-" prefix
                current_entity = {
                    "text": text[start:end],
                    "label": entity_type,
                    "start": start,
                    "end": end,
                    "confidence": confidence
                }
            
            elif label.startswith("I-") and current_entity:
                # Extend current entity
                entity_type = label[2:]  # Remove "I-" prefix
                
                # Only extend if same entity type
                if current_entity["label"] == entity_type:
                    current_entity["text"] = text[current_entity["start"]:end]
                    current_entity["end"] = end
                    # Update confidence (take minimum to be conservative)
                    current_entity["confidence"] = min(current_entity["confidence"], confidence)
                else:
                    # Different entity type, save previous and start new
                    entities.append(current_entity)
                    current_entity = {
                        "text": text[start:end],
                        "label": entity_type,
                        "start": start,
                        "end": end,
                        "confidence": confidence
                    }
            
            else:  # "O" label
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add last entity if exists
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def predict_batch(self, texts):
        """
        Predict NER for multiple texts
        
        Args:
            texts (list): List of texts
            
        Returns:
            list: List of predictions for each text
        """
        results = []
        for text in texts:
            entities = self.predict(text)
            results.append({
                "text": text,
                "entities": entities,
                "entity_count": len(entities)
            })
        return results
    
    def get_entity_counts(self, entities):
        """
        Count entities by type
        
        Args:
            entities: List of entities from predict()
            
        Returns:
            dict: Count per entity type
        """
        counts = {}
        for entity in entities:
            label = entity['label']
            counts[label] = counts.get(label, 0) + 1
        return counts


# ============================================================
# Singleton Pattern - Only load model once
# ============================================================
_model_instance = None

def get_model(model_path="./models/ner_models"):
    """
    Get or create model instance (singleton pattern)
    This ensures model is only loaded once
    
    Args:
        model_path: Path to model folder
        
    Returns:
        NERModel: Initialized model instance
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = NERModel(model_path)
    return _model_instance


# ============================================================
# For testing purposes
# ============================================================
if __name__ == "__main__":
    print("="*70)
    print("Testing NER Model")
    print("="*70)
    
    # Test load model
    try:
        model = get_model()
        
        # Test predict with multiple examples
        test_texts = [
            "Joko Widodo adalah presiden Indonesia yang tinggal di Jakarta.",
            "Apple Inc. membuka kantor baru di Singapura minggu lalu.",
            "Elon Musk adalah CEO Tesla dan SpaceX yang berbasis di Amerika Serikat."
        ]
        
        for i, test_text in enumerate(test_texts, 1):
            print(f"\n{'='*70}")
            print(f"📝 Test {i}: {test_text}")
            print(f"{'='*70}")
            print("🔍 Predictions:")
            
            entities = model.predict(test_text)
            
            if entities:
                for entity in entities:
                    print(f"   - {entity['text']:<25} → {entity['label']:<15} [{entity['start']:3}:{entity['end']:3}] (conf: {entity['confidence']:.3f})")
                
                # Show entity counts
                counts = model.get_entity_counts(entities)
                print(f"\n📊 Entity counts: {counts}")
            else:
                print("   No entities found")
        
        print("\n" + "="*70)
        print("✅ Test completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Make sure:")
        print("   1. Model files exist in ./models/ner_model/")
        print("   2. All required files are present (config.json, model weights, tokenizer files)")
        print("   3. The model is XLM-RoBERTa based on the config.json")