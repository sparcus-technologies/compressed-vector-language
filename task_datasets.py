"""
Task Dataset Generator for CVL Benchmarking
Generates datasets for 10 different task types to test CVL performance
"""

import random
import json
from typing import List, Dict, Tuple, Any
import os


class TaskDatasetGenerator:
    """Generate datasets for all 10 benchmark task types"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.task_generators = {
            'arithmetic': self.generate_arithmetic,
            'summarization': self.generate_summarization,
            'paraphrasing': self.generate_paraphrasing,
            'sentence_completion': self.generate_sentence_completion,
            'classification': self.generate_classification,
            'translation': self.generate_translation,
            'qa_factual': self.generate_qa_factual,
            'commonsense': self.generate_commonsense,
            'analogies': self.generate_analogies,
            'entity_extraction': self.generate_entity_extraction
        }
    
    # ==================== 1. ARITHMETIC / MATH WORD PROBLEMS ====================
    
    def generate_arithmetic(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate arithmetic and math word problems"""
        dataset = []
        operations = [
            ('+', 'plus', 'add'),
            ('-', 'minus', 'subtract'),
            ('*', 'times', 'multiply'),
            ('/', 'divided by', 'divide')
        ]
        
        for i in range(num_samples):
            a = random.randint(1, 99)
            b = random.randint(1, 99)
            op_symbol, op_word, op_verb = random.choice(operations)
            
            # Calculate answer
            if op_symbol == '+':
                answer = a + b
            elif op_symbol == '-':
                answer = a - b
            elif op_symbol == '*':
                answer = a * b
            else:  # division
                b = max(1, b)  # Avoid division by zero
                answer = round(a / b, 2)
            
            # Generate question (mix formats)
            question_formats = [
                f"What is {a} {op_word} {b}?",
                f"Calculate {a} {op_symbol} {b}",
                f"{a} {op_symbol} {b} = ?",
                f"If you {op_verb} {a} and {b}, what do you get?"
            ]
            question = random.choice(question_formats)
            
            dataset.append({
                'id': f'math_{i}',
                'question': question,
                'answer': str(answer),
                'answer_numeric': answer,
                'task_type': 'arithmetic',
                'difficulty': 'easy' if max(a, b) < 20 else 'medium' if max(a, b) < 50 else 'hard',
                'operation': op_symbol,
                'operands': [a, b]
            })
        
        return dataset
    
    # ==================== 2. SUMMARIZATION ====================
    
    def generate_summarization(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate summarization tasks with passages and expected summaries"""
        passages_and_summaries = [
            {
                'passage': "The Industrial Revolution was a period of major industrialization that took place during the late 1700s and early 1800s. It began in Great Britain and quickly spread throughout Western Europe and North America. This period saw the mechanization of agriculture and textile manufacturing and a revolution in power, including steam ships and railroads, that affected social, cultural, and economic conditions.",
                'summary': "The Industrial Revolution (late 1700s-early 1800s) began in Britain and spread globally, featuring mechanization of agriculture and textiles, plus revolutionary steam-powered transportation.",
                'keywords': ['Industrial Revolution', 'mechanization', 'steam power']
            },
            {
                'passage': "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil and gas, which produces heat-trapping greenhouse gases.",
                'summary': "Climate change involves long-term temperature and weather shifts, mainly driven by human fossil fuel burning since the 1800s, producing greenhouse gases.",
                'keywords': ['climate change', 'fossil fuels', 'greenhouse gases']
            },
            {
                'passage': "Artificial intelligence is the simulation of human intelligence by machines, especially computer systems. AI applications include expert systems, natural language processing, speech recognition and machine vision. AI programming focuses on three cognitive skills: learning, reasoning and self-correction.",
                'summary': "Artificial intelligence simulates human intelligence in machines through applications like NLP and vision, focusing on learning, reasoning, and self-correction.",
                'keywords': ['AI', 'machine learning', 'natural language processing']
            },
            {
                'passage': "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water. Photosynthesis in plants generally involves the green pigment chlorophyll and generates oxygen as a by-product.",
                'summary': "Photosynthesis is how plants use sunlight, chlorophyll, CO2, and water to create nutrients while producing oxygen.",
                'keywords': ['photosynthesis', 'chlorophyll', 'oxygen']
            },
            {
                'passage': "The human brain is the central organ of the human nervous system, and with the spinal cord makes up the central nervous system. The brain consists of the cerebrum, brainstem and cerebellum. It controls most of the activities of the body, processing, integrating, and coordinating information received from sense organs.",
                'summary': "The human brain, central to the nervous system, includes the cerebrum, brainstem, and cerebellum, controlling body activities and processing sensory information.",
                'keywords': ['brain', 'nervous system', 'cerebrum']
            },
            {
                'passage': "The Internet is a global system of interconnected computer networks that uses standard protocols to link billions of devices worldwide. It carries an extensive range of information resources and services, such as the World Wide Web, email, and file sharing.",
                'summary': "The Internet is a global network of interconnected computers using standard protocols to provide services like the web, email, and file sharing.",
                'keywords': ['Internet', 'network', 'World Wide Web']
            }
        ]
        
        dataset = []
        for i in range(num_samples):
            item = random.choice(passages_and_summaries)
            dataset.append({
                'id': f'summ_{i}',
                'passage': item['passage'],
                'expected_summary': item['summary'],
                'keywords': item['keywords'],
                'task_type': 'summarization',
                'max_words': 40,
                'passage_length': len(item['passage'].split())
            })
        
        return dataset
    
    # ==================== 3. PARAPHRASING / REWRITING ====================
    
    def generate_paraphrasing(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate paraphrasing and rewriting tasks"""
        sentence_pairs = [
            ("The cat sat on the mat.", "A feline rested on the floor covering."),
            ("It's raining heavily today.", "There's substantial precipitation falling currently."),
            ("She quickly ran to the store.", "She hurried rapidly to the shop."),
            ("The movie was incredibly boring.", "The film was extremely dull and uninteresting."),
            ("He enjoys playing basketball.", "He likes participating in basketball games."),
            ("The teacher explained the lesson clearly.", "The instructor clarified the material thoroughly."),
            ("We need to finish this project soon.", "We must complete this task in the near future."),
            ("The food tasted delicious.", "The meal had an excellent flavor."),
            ("She felt extremely happy.", "She experienced great joy and happiness."),
            ("The car moved very fast.", "The vehicle traveled at high speed."),
            ("They worked hard all day.", "They labored diligently throughout the entire day."),
            ("The book was interesting to read.", "The novel proved engaging and captivating."),
            ("He spoke softly to the child.", "He addressed the youngster in gentle tones."),
            ("The weather is quite cold today.", "The temperature is considerably low currently."),
            ("She has a beautiful voice.", "She possesses lovely vocal qualities.")
        ]
        
        dataset = []
        for i in range(num_samples):
            original, paraphrase = random.choice(sentence_pairs)
            
            # Randomly swap original and paraphrase for variety
            if random.random() < 0.3:
                original, paraphrase = paraphrase, original
            
            dataset.append({
                'id': f'para_{i}',
                'original': original,
                'expected_paraphrase': paraphrase,
                'task_type': 'paraphrasing',
                'original_length': len(original.split()),
                'paraphrase_length': len(paraphrase.split())
            })
        
        return dataset
    
    # ==================== 4. SENTENCE COMPLETION / CLOZE TESTS ====================
    
    def generate_sentence_completion(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate cloze test / sentence completion tasks"""
        templates = [
            ("The capital of France is [MASK].", "Paris", "geography"),
            ("Water boils at [MASK] degrees Celsius.", "100", "science"),
            ("The opposite of hot is [MASK].", "cold", "vocabulary"),
            ("A baby dog is called a [MASK].", "puppy", "animals"),
            ("The sun rises in the [MASK].", "east", "general knowledge"),
            ("There are [MASK] days in a week.", "seven", "general knowledge"),
            ("Birds can [MASK] in the sky.", "fly", "animals"),
            ("The color of grass is [MASK].", "green", "general knowledge"),
            ("The first month of the year is [MASK].", "January", "general knowledge"),
            ("A group of lions is called a [MASK].", "pride", "animals"),
            ("The largest ocean is the [MASK] Ocean.", "Pacific", "geography"),
            ("Plants need [MASK] to grow.", "sunlight", "science"),
            ("The capital of Japan is [MASK].", "Tokyo", "geography"),
            ("Ice melts into [MASK].", "water", "science"),
            ("A person who flies an airplane is a [MASK].", "pilot", "occupations"),
            ("The opposite of day is [MASK].", "night", "vocabulary"),
            ("Bees make [MASK].", "honey", "animals"),
            ("The planet closest to the sun is [MASK].", "Mercury", "science"),
            ("The Statue of Liberty is in [MASK] York.", "New", "geography"),
            ("Fire is [MASK].", "hot", "general knowledge")
        ]
        
        dataset = []
        for i in range(num_samples):
            sentence, answer, category = random.choice(templates)
            
            # Create alternative mask formats
            mask_formats = ["[MASK]", "____", "___?___", "[BLANK]"]
            masked_sentence = sentence.replace("[MASK]", random.choice(mask_formats))
            
            dataset.append({
                'id': f'cloze_{i}',
                'sentence': masked_sentence,
                'answer': answer,
                'category': category,
                'task_type': 'sentence_completion',
                'original_sentence': sentence
            })
        
        return dataset
    
    # ==================== 5. CLASSIFICATION ====================
    
    def generate_classification(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate text classification tasks (sentiment analysis)"""
        examples = [
            ("I love this product! It's absolutely amazing!", "positive", 0.95),
            ("This is the worst purchase I've ever made.", "negative", 0.90),
            ("The item is okay, nothing particularly special.", "neutral", 0.70),
            ("Absolutely fantastic service and quality!", "positive", 0.98),
            ("Terrible experience, would not recommend at all.", "negative", 0.92),
            ("It works as expected, no complaints.", "neutral", 0.75),
            ("Best decision I've made this year!", "positive", 0.96),
            ("Complete waste of money and time.", "negative", 0.94),
            ("Pretty decent, meets basic requirements.", "neutral", 0.72),
            ("Outstanding product, exceeded expectations!", "positive", 0.97),
            ("Very disappointing and poor quality.", "negative", 0.88),
            ("Average product, nothing wrong with it.", "neutral", 0.68),
            ("Highly recommend, five stars!", "positive", 0.99),
            ("Awful, broke after one use.", "negative", 0.91),
            ("It's fine for the price.", "neutral", 0.65),
            ("Incredible value and performance!", "positive", 0.93),
            ("Not worth it, many issues.", "negative", 0.85),
            ("Acceptable quality, does the job.", "neutral", 0.70),
            ("Love it! Will buy again!", "positive", 0.95),
            ("Very dissatisfied with this product.", "negative", 0.87)
        ]
        
        dataset = []
        for i in range(num_samples):
            text, sentiment, confidence = random.choice(examples)
            
            dataset.append({
                'id': f'class_{i}',
                'text': text,
                'label': sentiment,
                'confidence': confidence,
                'task_type': 'classification',
                'num_classes': 3,
                'classes': ['positive', 'negative', 'neutral'],
                'text_length': len(text.split())
            })
        
        return dataset
    
    # ==================== 6. TRANSLATION ====================
    
    def generate_translation(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate translation tasks between languages"""
        translations = [
            ("Hello", "Hola", "en", "es", "greeting"),
            ("Good morning", "Bonjour", "en", "fr", "greeting"),
            ("Thank you", "Danke", "en", "de", "courtesy"),
            ("Goodbye", "Adiós", "en", "es", "farewell"),
            ("How are you?", "Comment allez-vous?", "en", "fr", "conversation"),
            ("Please", "Por favor", "en", "es", "courtesy"),
            ("Yes", "Oui", "en", "fr", "response"),
            ("No", "Nein", "en", "de", "response"),
            ("I love you", "Te amo", "en", "es", "emotion"),
            ("Good night", "Bonne nuit", "en", "fr", "farewell"),
            ("Water", "Wasser", "en", "de", "noun"),
            ("Food", "Comida", "en", "es", "noun"),
            ("House", "Maison", "en", "fr", "noun"),
            ("Cat", "Katze", "en", "de", "animal"),
            ("Dog", "Perro", "en", "es", "animal"),
            ("Book", "Livre", "en", "fr", "noun"),
            ("Car", "Auto", "en", "de", "noun"),
            ("Friend", "Amigo", "en", "es", "noun"),
            ("Beautiful", "Belle", "en", "fr", "adjective"),
            ("Happy", "Glücklich", "en", "de", "adjective")
        ]
        
        dataset = []
        for i in range(num_samples):
            source, target, src_lang, tgt_lang, category = random.choice(translations)
            
            dataset.append({
                'id': f'trans_{i}',
                'source_text': source,
                'target_text': target,
                'source_lang': src_lang,
                'target_lang': tgt_lang,
                'category': category,
                'task_type': 'translation',
                'source_length': len(source.split()),
                'target_length': len(target.split())
            })
        
        return dataset
    
    # ==================== 7. QUESTION ANSWERING (FACTUAL RECALL) ====================
    
    def generate_qa_factual(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate factual question answering tasks"""
        qa_pairs = [
            ("Who wrote Pride and Prejudice?", "Jane Austen", "literature", "author"),
            ("What is the capital of Japan?", "Tokyo", "geography", "capital"),
            ("How many continents are there?", "7", "geography", "count"),
            ("What is the largest planet in our solar system?", "Jupiter", "astronomy", "planet"),
            ("Who painted the Mona Lisa?", "Leonardo da Vinci", "art", "artist"),
            ("What year did World War II end?", "1945", "history", "year"),
            ("What is the speed of light?", "299792458 meters per second", "physics", "constant"),
            ("Who was the first president of the United States?", "George Washington", "history", "president"),
            ("What is the chemical symbol for gold?", "Au", "chemistry", "symbol"),
            ("How many states are in the USA?", "50", "geography", "count"),
            ("Who invented the telephone?", "Alexander Graham Bell", "history", "inventor"),
            ("What is the capital of France?", "Paris", "geography", "capital"),
            ("What is the largest ocean?", "Pacific Ocean", "geography", "ocean"),
            ("Who wrote Romeo and Juliet?", "William Shakespeare", "literature", "author"),
            ("What is the smallest planet?", "Mercury", "astronomy", "planet"),
            ("What year did humans land on the moon?", "1969", "history", "year"),
            ("What is H2O?", "Water", "chemistry", "compound"),
            ("Who painted the Starry Night?", "Vincent van Gogh", "art", "artist"),
            ("What is the tallest mountain?", "Mount Everest", "geography", "mountain"),
            ("How many bones in the human body?", "206", "biology", "count")
        ]
        
        dataset = []
        for i in range(num_samples):
            question, answer, category, answer_type = random.choice(qa_pairs)
            
            dataset.append({
                'id': f'qa_{i}',
                'question': question,
                'answer': answer,
                'category': category,
                'answer_type': answer_type,
                'task_type': 'qa_factual',
                'question_length': len(question.split())
            })
        
        return dataset
    
    # ==================== 8. COMMONSENSE REASONING ====================
    
    def generate_commonsense(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate commonsense reasoning tasks"""
        questions = [
            ("Can a cat fly?", "No", "Cats don't have wings and cannot fly.", "animal abilities"),
            ("Is water wet?", "Yes", "Water is wet by definition.", "properties"),
            ("Can you eat a rock?", "No", "Rocks are inedible and would harm you.", "safety"),
            ("Does the sun come up in the morning?", "Yes", "The sun rises in the morning.", "nature"),
            ("Can fish breathe underwater?", "Yes", "Fish have gills to breathe underwater.", "animal abilities"),
            ("Do plants need sunlight?", "Yes", "Plants need sunlight for photosynthesis.", "nature"),
            ("Can humans breathe underwater without equipment?", "No", "Humans need air to breathe.", "human abilities"),
            ("Is ice cold?", "Yes", "Ice is frozen water and is cold.", "properties"),
            ("Can you walk through walls?", "No", "Walls are solid and block passage.", "physics"),
            ("Do dogs bark?", "Yes", "Dogs commonly bark as communication.", "animal behavior"),
            ("Can birds swim?", "Some can", "Some birds like ducks can swim.", "animal abilities"),
            ("Is fire hot?", "Yes", "Fire produces heat and is hot.", "properties"),
            ("Can you see in complete darkness?", "No", "Eyes need light to see.", "human abilities"),
            ("Do cars need fuel?", "Yes", "Cars need fuel or electricity to run.", "mechanics"),
            ("Can you eat without a mouth?", "No", "A mouth is required for eating.", "biology"),
            ("Is the sky blue?", "Usually", "The sky appears blue during daytime.", "nature"),
            ("Can trees walk?", "No", "Trees are rooted and cannot walk.", "nature"),
            ("Do you need oxygen to survive?", "Yes", "Humans need oxygen to breathe.", "biology"),
            ("Can you sleep with your eyes open?", "Difficult", "Most people need to close eyes to sleep.", "human abilities"),
            ("Is snow hot?", "No", "Snow is frozen water and is cold.", "properties")
        ]
        
        dataset = []
        for i in range(num_samples):
            question, answer, explanation, category = random.choice(questions)
            
            dataset.append({
                'id': f'cs_{i}',
                'question': question,
                'answer': answer,
                'explanation': explanation,
                'category': category,
                'task_type': 'commonsense',
                'question_length': len(question.split())
            })
        
        return dataset
    
    # ==================== 9. ANALOGIES / WORD RELATIONSHIPS ====================
    
    def generate_analogies(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate analogy and word relationship tasks"""
        analogies = [
            ("king", "queen", "man", "woman", "gender"),
            ("hot", "cold", "day", "night", "opposites"),
            ("dog", "puppy", "cat", "kitten", "young"),
            ("big", "small", "tall", "short", "opposites"),
            ("happy", "sad", "good", "bad", "opposites"),
            ("up", "down", "left", "right", "directions"),
            ("father", "mother", "son", "daughter", "family"),
            ("doctor", "hospital", "teacher", "school", "workplace"),
            ("bird", "fly", "fish", "swim", "actions"),
            ("hand", "glove", "foot", "shoe", "clothing"),
            ("car", "road", "boat", "water", "transportation"),
            ("pen", "write", "brush", "paint", "tools"),
            ("eye", "see", "ear", "hear", "senses"),
            ("cow", "milk", "chicken", "egg", "products"),
            ("tree", "forest", "star", "galaxy", "collections"),
            ("book", "read", "music", "listen", "activities"),
            ("winter", "cold", "summer", "hot", "seasons"),
            ("lion", "roar", "dog", "bark", "sounds"),
            ("rich", "poor", "strong", "weak", "opposites"),
            ("brain", "think", "heart", "pump", "functions")
        ]
        
        dataset = []
        for i in range(num_samples):
            a, b, c, d, category = random.choice(analogies)
            
            # Create question in standard format
            question = f"{a} is to {b} as {c} is to ?"
            
            dataset.append({
                'id': f'analog_{i}',
                'question': question,
                'word_a': a,
                'word_b': b,
                'word_c': c,
                'answer': d,
                'category': category,
                'task_type': 'analogies',
                'full_analogy': f"{a}:{b}::{c}:{d}"
            })
        
        return dataset
    
    # ==================== 10. ENTITY EXTRACTION ====================
    
    def generate_entity_extraction(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate named entity recognition / extraction tasks"""
        examples = [
            {
                'text': "Apple Inc. was founded by Steve Jobs in Cupertino on April 1, 1976.",
                'entities': {
                    'ORGANIZATION': ['Apple Inc.'],
                    'PERSON': ['Steve Jobs'],
                    'LOCATION': ['Cupertino'],
                    'DATE': ['April 1, 1976']
                }
            },
            {
                'text': "Barack Obama was born in Hawaii and served as the 44th President.",
                'entities': {
                    'PERSON': ['Barack Obama'],
                    'LOCATION': ['Hawaii'],
                    'POSITION': ['44th President']
                }
            },
            {
                'text': "Google was founded in September 1998 by Larry Page and Sergey Brin in California.",
                'entities': {
                    'ORGANIZATION': ['Google'],
                    'PERSON': ['Larry Page', 'Sergey Brin'],
                    'DATE': ['September 1998'],
                    'LOCATION': ['California']
                }
            },
            {
                'text': "The Eiffel Tower in Paris was completed in 1889 by Gustave Eiffel.",
                'entities': {
                    'LOCATION': ['Eiffel Tower', 'Paris'],
                    'DATE': ['1889'],
                    'PERSON': ['Gustave Eiffel']
                }
            },
            {
                'text': "Microsoft was established by Bill Gates and Paul Allen in April 1975.",
                'entities': {
                    'ORGANIZATION': ['Microsoft'],
                    'PERSON': ['Bill Gates', 'Paul Allen'],
                    'DATE': ['April 1975']
                }
            },
            {
                'text': "The United Nations was founded on October 24, 1945, in San Francisco.",
                'entities': {
                    'ORGANIZATION': ['United Nations'],
                    'DATE': ['October 24, 1945'],
                    'LOCATION': ['San Francisco']
                }
            },
            {
                'text': "Albert Einstein published his theory of relativity in 1905 in Germany.",
                'entities': {
                    'PERSON': ['Albert Einstein'],
                    'DATE': ['1905'],
                    'LOCATION': ['Germany']
                }
            },
            {
                'text': "Amazon was started by Jeff Bezos in Seattle in July 1994.",
                'entities': {
                    'ORGANIZATION': ['Amazon'],
                    'PERSON': ['Jeff Bezos'],
                    'LOCATION': ['Seattle'],
                    'DATE': ['July 1994']
                }
            }
        ]
        
        dataset = []
        for i in range(num_samples):
            item = random.choice(examples)
            
            # Count total entities
            total_entities = sum(len(entities) for entities in item['entities'].values())
            
            dataset.append({
                'id': f'ner_{i}',
                'text': item['text'],
                'entities': item['entities'],
                'task_type': 'entity_extraction',
                'text_length': len(item['text'].split()),
                'entity_count': total_entities,
                'entity_types': list(item['entities'].keys())
            })
        
        return dataset
    
    # ==================== UTILITY METHODS ====================
    
    def generate_all_tasks(self, samples_per_task: int = 50) -> Dict[str, List[Dict]]:
        """Generate all task datasets at once"""
        print("=" * 70)
        print("GENERATING BENCHMARK DATASETS FOR 10 TASK TYPES")
        print("=" * 70)
        
        all_datasets = {}
        
        for i, (task_name, generator_func) in enumerate(self.task_generators.items(), 1):
            print(f"[{i}/10] Generating {task_name}... ", end='')
            all_datasets[task_name] = generator_func(samples_per_task)
            print(f"✓ ({len(all_datasets[task_name])} samples)")
        
        print(f"\n✓ Successfully generated {len(all_datasets)} task types")
        print(f"  Total samples: {sum(len(d) for d in all_datasets.values())}")
        
        return all_datasets
    
    def save_datasets(self, filepath: str = "benchmark_datasets.json"):
        """Save all datasets to JSON file"""
        datasets = self.generate_all_tasks()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(datasets, f, indent=2, ensure_ascii=False)
        
        file_size = os.path.getsize(filepath) / 1024  # KB
        print(f"\n✓ Saved all datasets to {filepath} ({file_size:.1f} KB)")
        
        return datasets
    
    def get_dataset_statistics(self, datasets: Dict[str, List[Dict]] = None) -> Dict[str, Any]:
        """Get comprehensive statistics about the datasets"""
        if datasets is None:
            datasets = self.generate_all_tasks(50)
        
        stats = {
            'total_tasks': len(datasets),
            'total_samples': sum(len(d) for d in datasets.values()),
            'task_breakdown': {},
            'avg_samples_per_task': 0
        }
        
        for task_name, task_data in datasets.items():
            stats['task_breakdown'][task_name] = {
                'count': len(task_data),
                'sample_keys': list(task_data[0].keys()) if task_data else []
            }
        
        stats['avg_samples_per_task'] = stats['total_samples'] / stats['total_tasks']
        
        return stats


def main():
    """Demonstrate task dataset generation"""
    print("=" * 70)
    print("TASK DATASET GENERATOR DEMONSTRATION")
    print("=" * 70)
    
    generator = TaskDatasetGenerator()
    datasets = generator.generate_all_tasks(samples_per_task=30)
    
    # Print summary
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    
    for task_name, task_data in datasets.items():
        print(f"\n{task_name.upper().replace('_', ' ')}:")
        print(f"  Total samples: {len(task_data)}")
        
        # Show example
        if task_data:
            example = task_data[0]
            print(f"  Example keys: {list(example.keys())}")
            
            # Show first example based on task type
            if 'question' in example:
                print(f"  Sample Q: {example['question'][:60]}...")
                print(f"  Sample A: {example.get('answer', 'N/A')}")
            elif 'text' in example:
                print(f"  Sample: {example['text'][:60]}...")
            elif 'passage' in example:
                print(f"  Sample: {example['passage'][:60]}...")
    
    # Get statistics
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    
    stats = generator.get_dataset_statistics(datasets)
    print(f"Total task types: {stats['total_tasks']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Avg samples per task: {stats['avg_samples_per_task']:.1f}")
    
    # Save to file
    print("\n" + "=" * 70)
    generator.save_datasets("benchmark_datasets.json")
    
    print("\n✓ Task dataset generation complete!")


if __name__ == "__main__":
    main()

