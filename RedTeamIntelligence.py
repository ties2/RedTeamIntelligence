
import pandas as pd
import numpy as np
import networkx as nx
import uuid
import json
import logging
import os
import pickle
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import yaml
import joblib
import sys

# Check for TensorFlow availability
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, save_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not found. LSTM sequence model will be disabled. Install TensorFlow with: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'red_team_{datetime.now().strftime("%Y%m%d")}.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

class MITREIntegration:
    """Class to fetch and integrate MITRE ATT&CK data"""
    
    def __init__(self, use_local_cache=True):
        self.use_local_cache = use_local_cache
        self.cache_file = 'mitre_data_cache.pkl'
        self.mitre_enterprise_url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
        self.mitre_data = self._load_mitre_data()
        
    def _load_mitre_data(self):
        """Load MITRE data from cache or fetch from source"""
        if self.use_local_cache and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    logging.info("Loading MITRE data from cache")
                    return pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading MITRE cache: {e}")
        
        try:
            logging.info("Fetching MITRE ATT&CK data")
            response = requests.get(self.mitre_enterprise_url)
            data = response.json()
            
            # Cache the data
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            return data
        except Exception as e:
            logging.error(f"Error fetching MITRE data: {e}")
            # Return an empty dict if fetching fails
            return {"objects": []}
    
    def get_techniques_by_tactic(self, tactic_name: str) -> List[Dict]:
        """Get techniques associated with a specific tactic"""
        techniques = []
        tactic_id = None
        
        # First, find the tactic ID
        for obj in self.mitre_data.get("objects", []):
            if obj.get("type") == "x-mitre-tactic" and obj.get("name", "").lower() == tactic_name.lower():
                tactic_id = obj.get("id")
                break
        
        if not tactic_id:
            return []
            
        # Then find techniques that use this tactic
        for obj in self.mitre_data.get("objects", []):
            if obj.get("type") == "attack-pattern":
                if "kill_chain_phases" in obj:
                    for phase in obj["kill_chain_phases"]:
                        if phase.get("phase_id") == tactic_id:
                            techniques.append({
                                "id": obj.get("external_references", [{}])[0].get("external_id", ""),
                                "name": obj.get("name", ""),
                                "description": obj.get("description", "")
                            })
        
        return techniques
    
    def get_apt_groups(self) -> List[Dict]:
        """Get all APT groups from MITRE"""
        groups = []
        
        for obj in self.mitre_data.get("objects", []):
            if obj.get("type") == "intrusion-set":
                groups.append({
                    "id": obj.get("external_references", [{}])[0].get("external_id", ""),
                    "name": obj.get("name", ""),
                    "description": obj.get("description", "")
                })
        
        return groups
    
    def get_group_techniques(self, group_name: str) -> List[Dict]:
        """Get techniques associated with a specific APT group"""
        group_id = None
        techniques = []
        
        # Find the group ID
        for obj in self.mitre_data.get("objects", []):
            if obj.get("type") == "intrusion-set" and obj.get("name", "").lower() == group_name.lower():
                group_id = obj.get("id")
                break
        
        if not group_id:
            return []
        
        # Find relationships between this group and techniques
        for obj in self.mitre_data.get("objects", []):
            if obj.get("type") == "relationship" and obj.get("source_ref") == group_id and obj.get("relationship_type") == "uses":
                target_ref = obj.get("target_ref")
                
                # Find the technique this relationship points to
                for tech_obj in self.mitre_data.get("objects", []):
                    if tech_obj.get("id") == target_ref and tech_obj.get("type") == "attack-pattern":
                        techniques.append({
                            "id": tech_obj.get("external_references", [{}])[0].get("external_id", ""),
                            "name": tech_obj.get("name", ""),
                            "description": tech_obj.get("description", "")
                        })
        
        return techniques

class TTPDatabase:
    """Enhanced TTP Database with MITRE ATT&CK alignment and graph structure"""
    
    def __init__(self, mitre_integration=None):
        self.ttps = []
        self.graph = nx.DiGraph()
        self.mitre = mitre_integration if mitre_integration else MITREIntegration()
        self.label_encoder = LabelEncoder()
        self._load_data()
    
    def _load_data(self):
        """Load existing data if available"""
        if os.path.exists('ttp_database.json'):
            try:
                with open('ttp_database.json', 'r') as f:
                    self.ttps = json.load(f)
                    # Rebuild graph
                    for ttp in self.ttps:
                        self.graph.add_node(ttp["id"], **ttp)
                logging.info(f"Loaded {len(self.ttps)} TTPs from database")
                
                # Load relationships
                if os.path.exists('ttp_relationships.json'):
                    with open('ttp_relationships.json', 'r') as f:
                        relationships = json.load(f)
                        for rel in relationships:
                            self.graph.add_edge(
                                rel["source"], 
                                rel["target"], 
                                type=rel["type"],
                                probability=rel.get("probability", 1.0)
                            )
                    logging.info(f"Loaded {len(relationships)} relationships")
            except Exception as e:
                logging.error(f"Error loading TTP database: {e}")
    
    def save_data(self):
        """Save TTP database to file"""
        try:
            with open('ttp_database.json', 'w') as f:
                json.dump(self.ttps, f, indent=2)
            
            # Save relationships
            relationships = []
            for u, v, data in self.graph.edges(data=True):
                relationships.append({
                    "source": u,
                    "target": v,
                    "type": data.get("type"),
                    "probability": data.get("probability", 1.0)
                })
            
            with open('ttp_relationships.json', 'w') as f:
                json.dump(relationships, f, indent=2)
                
            logging.info(f"Saved {len(self.ttps)} TTPs and {len(relationships)} relationships to database")
        except Exception as e:
            logging.error(f"Error saving TTP database: {e}")
    
    def add_ttp(self, tactic, technique, apt_group, skills_required, tools_targeted, mitre_id, 
                detection_difficulty=None, success_probability=None, description=None):
        """Add a TTP to the database with extended attributes"""
        ttp_id = str(uuid.uuid4())
        
        # Get more details from MITRE if available
        mitre_details = {}
        techniques = self.mitre.get_techniques_by_tactic(tactic)
        for tech in techniques:
            if tech["id"] == mitre_id or tech["name"].lower() == technique.lower():
                mitre_details = tech
                break
        
        ttp = {
            "id": ttp_id,
            "tactic": tactic,
            "technique": technique,
            "apt_group": apt_group,
            "skills_required": skills_required,
            "tools_targeted": tools_targeted,
            "mitre_id": mitre_id,
            "detection_difficulty": detection_difficulty or np.random.uniform(0.2, 0.9),
            "success_probability": success_probability or np.random.uniform(0.3, 0.95),
            "description": description or mitre_details.get("description", f"Implementation of {technique}"),
            "timestamp": datetime.now().isoformat(),
            "mitre_details": mitre_details
        }
        
        self.ttps.append(ttp)
        self.graph.add_node(ttp_id, **ttp)
        logging.info(f"Added TTP: {ttp_id} - {tactic}:{technique}")
        
        # Auto-save after adding
        self.save_data()
        return ttp_id
    
    def add_relationship(self, ttp_id1, ttp_id2, relationship_type, probability=1.0):
        """Add a directed relationship between TTPs with probability"""
        self.graph.add_edge(ttp_id1, ttp_id2, type=relationship_type, probability=probability)
        logging.info(f"Added relationship: {ttp_id1} -> {ttp_id2} ({relationship_type}, prob={probability:.2f})")
        
        # Auto-save after adding
        self.save_data()
    
    def to_dataframe(self):
        """Convert TTPs to pandas DataFrame"""
        if not self.ttps:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.ttps)
        
        # Flatten skills and tools columns
        df['skills_count'] = df['skills_required'].apply(len)
        df['tools_count'] = df['tools_targeted'].apply(len)
        
        # Encode categorical columns
        if len(df) > 0:
            self.label_encoder.fit(df['tactic'])
            
        return df
    
    def get_ttp(self, ttp_id):
        """Get a TTP by ID"""
        return next((ttp for ttp in self.ttps if ttp["id"] == ttp_id), None)
    
    def get_ttps_by_apt(self, apt_group):
        """Get all TTPs associated with an APT group"""
        return [ttp for ttp in self.ttps if ttp["apt_group"] == apt_group]
    
    def get_ttps_by_tactic(self, tactic):
        """Get all TTPs for a specific tactic"""
        return [ttp for ttp in self.ttps if ttp["tactic"] == tactic]
    
    def get_attack_chain(self, start_ttp_id, max_length=5):
        """Get a realistic attack chain starting from a specific TTP"""
        if start_ttp_id not in self.graph:
            return []
            
        chain = [self.get_ttp(start_ttp_id)]
        current_id = start_ttp_id
        
        # Generate a path through the graph based on edge probabilities
        for _ in range(max_length - 1):
            if not list(self.graph.successors(current_id)):
                break
                
            # Choose next TTP based on edge probabilities
            successors = list(self.graph.successors(current_id))
            if not successors:
                break
                
            # Get probabilities for each successor
            probs = [self.graph[current_id][succ].get('probability', 1.0) for succ in successors]
            probs = np.array(probs) / sum(probs)  # Normalize
            
            next_id = np.random.choice(successors, p=probs)
            chain.append(self.get_ttp(next_id))
            current_id = next_id
            
        return chain
    
    def visualize_graph(self, output_file='ttp_graph.png'):
        """Visualize the TTP graph"""
        plt.figure(figsize=(12, 8))
        
        # Create position layout
        pos = nx.spring_layout(self.graph)
        
        # Get node colors by tactic
        tactics = [self.graph.nodes[n].get('tactic', 'Unknown') for n in self.graph.nodes()]
        unique_tactics = list(set(tactics))
        color_map = plt.cm.tab10(np.linspace(0, 1, len(unique_tactics)))
        node_colors = [color_map[unique_tactics.index(t)] for t in tactics]
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=500, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5, arrows=True)
        
        # Draw labels
        node_labels = {n: f"{self.graph.nodes[n].get('technique', 'Unknown')}" for n in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=8)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                        label=tactic, markerfacecolor=color_map[i], markersize=10)
                        for i, tactic in enumerate(unique_tactics)]
        plt.legend(handles=legend_elements, title='Tactics')
        
        plt.title('TTP Relationship Graph')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        logging.info(f"Graph visualization saved to {output_file}")
        return output_file

class APTProfiler:
    """APT Group profiling and simulation"""
    
    def __init__(self, mitre_integration=None):
        self.mitre = mitre_integration if mitre_integration else MITREIntegration()
        self.apt_profiles = {}
        self.load_apt_profiles()
    
    def load_apt_profiles(self):
        """Load APT profiles from file or initialize from MITRE"""
        if os.path.exists('apt_profiles.json'):
            try:
                with open('apt_profiles.json', 'r') as f:
                    self.apt_profiles = json.load(f)
                logging.info(f"Loaded {len(self.apt_profiles)} APT profiles")
            except Exception as e:
                logging.error(f"Error loading APT profiles: {e}")
                self.initialize_apt_profiles()
        else:
            self.initialize_apt_profiles()
    
    def initialize_apt_profiles(self):
        """Initialize APT profiles from MITRE data"""
        groups = self.mitre.get_apt_groups()
        
        for group in groups:
            techniques = self.mitre.get_group_techniques(group["name"])
            
            # Count techniques by tactic to get preferences
            tactic_counts = {}
            for tech in techniques:
                tactic = self._get_tactic_for_technique(tech["id"])
                if tactic:
                    tactic_counts[tactic] = tactic_counts.get(tactic, 0) + 1
            
            # Calculate preferred tactics
            total = sum(tactic_counts.values()) if tactic_counts else 1
            tactic_preferences = {tactic: count/total for tactic, count in tactic_counts.items()}
            
            # Create profile
            self.apt_profiles[group["name"]] = {
                "id": group["id"],
                "name": group["name"],
                "description": group["description"],
                "tactic_preferences": tactic_preferences,
                "known_techniques": [t["id"] for t in techniques],
                "skill_level": np.random.uniform(0.5, 0.95),  # Random initial skill level
                "persistence": np.random.uniform(0.3, 0.9),   # How persistent they are
                "stealth": np.random.uniform(0.3, 0.9),       # How stealthy they are
                "creativity": np.random.uniform(0.2, 0.8)     # How creative they are in developing new TTPs
            }
        
        self.save_apt_profiles()
        logging.info(f"Initialized {len(self.apt_profiles)} APT profiles")
    
    def _get_tactic_for_technique(self, technique_id):
        """Map a technique ID to its primary tactic"""
        # Simplified mapping for common techniques
        technique_tactic_map = {
            "T1595": "reconnaissance",
            "T1190": "initial-access",
            "T1566": "initial-access",
            "T1078": "persistence",
            "T1547": "persistence",
            "T1053": "privilege-escalation",
            "T1055": "defense-evasion",
            "T1027": "defense-evasion",
            "T1087": "discovery",
            "T1021": "lateral-movement",
            "T1569": "execution",
            "T1213": "collection",
            "T1001": "command-and-control",
            "T1048": "exfiltration",
            "T1485": "impact"
        }
        
        # Check if we have a direct mapping
        if technique_id in technique_tactic_map:
            return technique_tactic_map[technique_id]
        
        # Check if it's a sub-technique
        base_id = technique_id.split('.')[0] if '.' in technique_id else technique_id
        if base_id in technique_tactic_map:
            return technique_tactic_map[base_id]
        
        return None
    
    def save_apt_profiles(self):
        """Save APT profiles to file"""
        with open('apt_profiles.json', 'w') as f:
            json.dump(self.apt_profiles, f, indent=2)
        logging.info(f"Saved {len(self.apt_profiles)} APT profiles")
    
    def get_apt_profile(self, apt_name):
        """Get an APT profile by name"""
        return self.apt_profiles.get(apt_name)
    
    def get_all_apt_names(self):
        """Get a list of all APT group names"""
        return list(self.apt_profiles.keys())
    
    def generate_apt_ttp(self, db, tactic, apt_name):
        """Generate a TTP based on APT profile"""
        profile = self.get_apt_profile(apt_name)
        if not profile:
            logging.warning(f"No profile found for {apt_name}, using default parameters")
            creativity = 0.5
            skill_level = 0.7
        else:
            creativity = profile.get("creativity", 0.5)
            skill_level = profile.get("skill_level", 0.7)
        
        # Get techniques for this tactic
        techniques = self.mitre.get_techniques_by_tactic(tactic)
        
        if not techniques:
            # Fallback to common techniques
            base_techniques = [
                "Active Scanning", "Phishing", "Drive-by Compromise", 
                "Registry Run Keys", "Scheduled Task", "Process Injection"
            ]
            base_technique = np.random.choice(base_techniques)
        else:
            # Choose a technique based on APT profile preferences
            technique_obj = np.random.choice(techniques)
            base_technique = technique_obj["name"]
            mitre_id = technique_obj["id"]
        
        # Apply creativity based on APT profile
        if np.random.random() < creativity:
            creativity_factor = np.random.choice(["novel", "modified", "combined"])
            if creativity_factor == "novel":
                new_technique = f"Novel_{base_technique}_{uuid.uuid4().hex[:8]}"
            elif creativity_factor == "combined":
                new_technique = f"Combined_{base_technique}_With_{np.random.choice(['Phishing', 'Exfiltration', 'Encryption', 'Process Hollowing'])}"
            else:
                new_technique = f"{base_technique}_Modified_{uuid.uuid4().hex[:6]}"
        else:
            new_technique = base_technique
        
        # Skills required based on APT skill level
        all_skills = [
            "computer_architecture", "networking", "programming", 
            "creative_thinking", "reverse_engineering", "exploit_development",
            "social_engineering", "infrastructure_management", "cryptography"
        ]
        
        # More skilled APTs have more skills
        num_skills = max(2, int(skill_level * 7))
        skills = np.random.choice(all_skills, size=min(num_skills, len(all_skills)), replace=False).tolist()
        
        # Tools targeted based on the technique
        all_tools = ["EDR", "XDR", "SIEM", "Antivirus", "Firewall", "DLP", "IDS/IPS", "SOAR", "IAM"]
        num_tools = max(1, int((1 - skill_level) * 5))  # More skilled APTs target fewer tools (evade more)
        tools = np.random.choice(all_tools, size=min(num_tools, len(all_tools)), replace=False).tolist()
        
        # Calculate detection difficulty based on APT stealth level
        stealth = profile.get("stealth", 0.5) if profile else 0.5
        detection_difficulty = np.random.beta(stealth * 10, (1 - stealth) * 5)
        
        # Calculate success probability based on skill level
        success_probability = np.random.beta(skill_level * 10, (1 - skill_level) * 5)
        
        # Add TTP to database
        ttp_id = db.add_ttp(
            tactic=tactic,
            technique=new_technique,
            apt_group=apt_name,
            skills_required=skills,
            tools_targeted=tools,
            mitre_id=mitre_id if 'mitre_id' in locals() else f"T{np.random.randint(1000, 9999)}",
            detection_difficulty=detection_difficulty,
            success_probability=success_probability,
            description=f"APT {apt_name} implementation of {new_technique} for {tactic} phase"
        )
        
        logging.info(f"Generated APT TTP: {new_technique} for {apt_name}")
        return ttp_id

class FeatureExtractor:
    """Extract features from TTPs for ML model training"""
    
    def __init__(self, db):
        self.db = db
        self.tactic_encoder = LabelEncoder()
        self.apt_encoder = LabelEncoder()
        self.technique_vectorizer = None
        
        # Initialize encoders if we have data
        df = self.db.to_dataframe()
        if not df.empty:
            self.tactic_encoder.fit(df['tactic'])
            self.apt_encoder.fit(df['apt_group'])
            
            # Initialize technique vectorizer (placeholder)
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.technique_vectorizer = TfidfVectorizer(max_features=50)
            self.technique_vectorizer.fit(df['technique'])
    
    def extract_features(self, ttp_list=None):
        """Extract features from TTPs for ML model"""
        if ttp_list is None:
            df = self.db.to_dataframe()
            if df.empty:
                return None, None
            ttp_list = self.db.ttps
        else:
            df = pd.DataFrame(ttp_list)
        
        # Re-fit encoders if necessary
        self.tactic_encoder.fit(df['tactic'])
        self.apt_encoder.fit(df['apt_group'])
        
        features = []
        labels = []
        
        for ttp in ttp_list:
            tactic = ttp["tactic"]
            technique = ttp["technique"]
            skills = ttp["skills_required"]
            tools = ttp["tools_targeted"]
            apt_group = ttp["apt_group"]
            
            # Basic features
            feature_vector = [
                len(technique),
                len(skills),
                len(tools),
                ttp.get("detection_difficulty", 0.5),
                ttp.get("success_probability", 0.5)
            ]
            
            # Add encoded APT group
            apt_encoded = self.apt_encoder.transform([apt_group])[0]
            feature_vector.append(apt_encoded)
            
            # Add skill flags
            all_skills = [
                "computer_architecture", "networking", "programming", 
                "creative_thinking", "reverse_engineering", "exploit_development",
                "social_engineering", "infrastructure_management", "cryptography"
            ]
            for skill in all_skills:
                feature_vector.append(1 if skill in skills else 0)
            
            # Add tool flags
            all_tools = ["EDR", "XDR", "SIEM", "Antivirus", "Firewall", "DLP", "IDS/IPS", "SOAR", "IAM"]
            for tool in all_tools:
                feature_vector.append(1 if tool in tools else 0)
            
            # Add encoded tactic as the label
            tactic_encoded = self.tactic_encoder.transform([tactic])[0]
            
            features.append(feature_vector)
            labels.append(tactic_encoded)
        
        return np.array(features), np.array(labels)
    
    def extract_sequence_features(self, attack_chains):
        """Extract features for sequence model from attack chains"""
        if not TENSORFLOW_AVAILABLE:
            return None, None
            
        X_seq = []
        y_seq = []
        
        for chain in attack_chains:
            if len(chain) < 2:
                continue
                
            # Extract features for each TTP in the chain
            chain_features = []
            for ttp in chain[:-1]:  # All but the last one
                tactic = ttp["tactic"]
                technique = ttp["technique"]
                
                # Simple features for sequences
                feature_vector = [
                    len(technique),
                    len(ttp["skills_required"]),
                    len(ttp["tools_targeted"]),
                    ttp.get("detection_difficulty", 0.5),
                    ttp.get("success_probability", 0.5),
                    self.apt_encoder.transform([ttp["apt_group"]])[0],
                    self.tactic_encoder.transform([tactic])[0]
                ]
                chain_features.append(feature_vector)
            
            # Pad sequence if needed
            if len(chain_features) < 10:  # Max sequence length
                chain_features = pad_sequences([chain_features], maxlen=10, padding='post', dtype='float32')[0]
            
            # Target is the tactic of the last TTP
            target = self.tactic_encoder.transform([chain[-1]["tactic"]])[0]
            
            X_seq.append(chain_features)
            y_seq.append(target)
        
        return np.array(X_seq), np.array(y_seq)

class RedTeamML:
    """Enhanced ML Model for TTP classification and sequence prediction"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.sequence_model = self._build_lstm_model() if TENSORFLOW_AVAILABLE else None
        self.feature_extractor = None
        
        self.is_classifier_trained = False
        self.is_sequence_trained = False
        
        # Try to load existing models
        self._load_models()
    
    def _build_lstm_model(self):
        """Build LSTM model for attack sequence prediction"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = Sequential([
            LSTM(64, input_shape=(10, 7), return_sequences=True),  # 10 timesteps, 7 features
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(8, activation='softmax')  # Assuming 8 tactics (can be changed)
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def _load_models(self):
        """Load trained models if available"""
        # Load classifier
        clf_path = os.path.join(self.model_dir, 'classifier_model.joblib')
        if os.path.exists(clf_path):
            try:
                self.classifier = joblib.load(clf_path)
                self.is_classifier_trained = True
                logging.info("Loaded classifier model")
            except Exception as e:
                logging.error(f"Error loading classifier model: {e}")
        
        # Load sequence model
        seq_path = os.path.join(self.model_dir, 'sequence_model')
        if TENSORFLOW_AVAILABLE and os.path.exists(seq_path):
            try:
                self.sequence_model = load_model(seq_path)
                self.is_sequence_trained = True
                logging.info("Loaded sequence model")
            except Exception as e:
                logging.error(f"Error loading sequence model: {e}")
    

    def _save_models(self):
        """Save trained models"""
        # Save classifier
        clf_path = os.path.join(self.model_dir, 'classifier_model.joblib')
        try:
            joblib.dump(self.classifier, clf_path)
            logging.info(f"Saved classifier model to {clf_path}")
        except Exception as e:
            logging.error(f"Error saving classifier model: {e}")
        
        # Save sequence model
        if TENSORFLOW_AVAILABLE and self.is_sequence_trained:
            seq_path = os.path.join(self.model_dir, 'sequence_model')
            try:
                self.sequence_model.save(seq_path)
                logging.info(f"Saved sequence model to {seq_path}")
            except Exception as e:
                logging.error(f"Error saving sequence model: {e}")
    
    def set_feature_extractor(self, extractor):
        """Set the feature extractor"""
        self.feature_extractor = extractor
    
    def train_classifier(self, X, y):
        """Train the TTP classifier with hyperparameter tuning"""
        if len(np.unique(y)) < 2:
            logging.warning("Need at least two tactics for training.")
            return
        
        logging.info("Training classifier model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Grid search for hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
        # Use a smaller grid if we have little data
        if len(X) < 20:
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10]
            }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=min(5, len(np.unique(y))),
            scoring='accuracy',
            n_jobs=-1 if len(X) > 10 else 1
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search.fit(X_train, y_train)
        
        self.classifier = grid_search.best_estimator_
        self.is_classifier_trained = True
        
        # Evaluate model
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"Classifier trained with accuracy: {accuracy:.2f}")
        logging.info(f"Best parameters: {grid_search.best_params_}")
        
        if len(np.unique(y)) > 2:  # Only create report for multi-class
            report = classification_report(y_test, y_pred)
            logging.info(f"Classification report:\n{report}")
        
        # Save the model
        self._save_models()
        
        return accuracy
    
    def train_sequence_model(self, X_seq, y_seq, epochs=20):
        """Train LSTM sequence model with early stopping"""
        if not TENSORFLOW_AVAILABLE or self.sequence_model is None:
            logging.warning("Sequence model training skipped: TensorFlow not available.")
            return None
        
        if len(X_seq) < 10:
            logging.warning("Not enough sequence data for training. Need at least 10 sequences.")
            return None
        
        logging.info("Training sequence model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
        
        # Callbacks for early stopping and checkpointing
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(
            os.path.join(self.model_dir, 'sequence_model_best.h5'),
            monitor='val_loss',
            save_best_only=True
        )
        
        # Train the model
        history = self.sequence_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=min(32, len(X_train)),
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        # Evaluate
        loss, accuracy = self.sequence_model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"Sequence model trained with accuracy: {accuracy:.2f}")
        
        self.is_sequence_trained = True
        self._save_models()
        
        # Plot training history
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'sequence_training_history.png'))
        plt.close()
        
        return accuracy
    
    def predict_tactic(self, X, return_proba=False):
        """Predict tactic from feature vector"""
        if not self.is_classifier_trained:
            raise ValueError("Classifier not trained yet.")
        
        if return_proba:
            return self.classifier.predict_proba(X)
        return self.classifier.predict(X)
    
    def predict_attack_chain(self, initial_ttp_features, chain_length=3):
        """Predict a likely attack chain from initial TTP features"""
        if not TENSORFLOW_AVAILABLE or not self.is_sequence_trained:
            logging.warning("Sequence model prediction unavailable: TensorFlow not installed or model not trained.")
            return None
        
        if self.feature_extractor is None:
            logging.error("Feature extractor not set")
            return None
        
        # Generate predicted tactics for each step
        sequence = [initial_ttp_features]
        all_tactics = self.feature_extractor.tactic_encoder.classes_
        
        for _ in range(chain_length - 1):
            # Prepare sequence for prediction
            padded_seq = pad_sequences([sequence], maxlen=10, padding='post', dtype='float32')
            
            # Predict next tactic
            pred = self.sequence_model.predict(padded_seq, verbose=0)
            next_tactic_idx = np.argmax(pred[0])
            
            # Convert to tactic name
            if next_tactic_idx < len(all_tactics):
                next_tactic = all_tactics[next_tactic_idx]
                
                # Create a dummy feature vector for the next tactic
                next_features = initial_ttp_features.copy()
                next_features[-1] = next_tactic_idx  # Update tactic index
                
                sequence.append(next_features)
            else:
                break
        
        # Convert sequence indices back to tactic names
        return [all_tactics[int(seq[-1])] for seq in sequence]


class SecurityToolEvaluator:
    """Evaluate security tools against TTPs"""
    
    def __init__(self, results_dir='evaluation_results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.evaluations = self._load_evaluations()
    
    def _load_evaluations(self):
        """Load previous evaluations"""
        eval_file = os.path.join(self.results_dir, 'evaluations.json')
        if os.path.exists(eval_file):
            try:
                with open(eval_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading evaluations: {e}")
        return []
    
    def _save_evaluations(self):
        """Save evaluations to file"""
        eval_file = os.path.join(self.results_dir, 'evaluations.json')
        try:
            with open(eval_file, 'w') as f:
                json.dump(self.evaluations, f, indent=2)
            logging.info(f"Saved {len(self.evaluations)} evaluations")
        except Exception as e:
            logging.error(f"Error saving evaluations: {e}")
    
    def evaluate_ttp(self, ttp, security_tools=None):
        """Evaluate a TTP against security tools"""
        if security_tools is None:
            security_tools = ttp["tools_targeted"]
        
        # Calculate detection probabilities based on TTP characteristics
        base_detection_prob = 1.0 - ttp.get("detection_difficulty", 0.5)
        
        tool_evaluations = {}
        detected_by = []
        
        for tool in security_tools:
            # Simulate tool-specific detection probability
            if tool == "EDR":
                # EDRs are better at detecting execution techniques
                detect_prob = base_detection_prob * 1.2 if "execution" in ttp["tactic"].lower() else base_detection_prob
            elif tool == "SIEM":
                # SIEMs are better at detecting unusual traffic patterns
                detect_prob = base_detection_prob * 1.3 if "command-and-control" in ttp["tactic"].lower() else base_detection_prob
            elif tool == "Firewall":
                # Firewalls detect network-based attacks better
                detect_prob = base_detection_prob * 1.4 if "lateral-movement" in ttp["tactic"].lower() else base_detection_prob * 0.8
            else:
                detect_prob = base_detection_prob
            
            # Cap probability between 0.1 and 0.95
            detect_prob = max(0.1, min(0.95, detect_prob))
            
            # Simulate detection outcome
            detected = np.random.random() < detect_prob
            if detected:
                detected_by.append(tool)
            
            tool_evaluations[tool] = {
                "detection_probability": float(detect_prob),
                "detected": bool(detected),
                "detection_signature": f"{tool}_sig_{uuid.uuid4().hex[:8]}" if detected else None,
                "evasion_difficulty": float(np.random.uniform(0.4, 0.9))
            }
        
        # Create evaluation record
        evaluation = {
            "ttp_id": ttp["id"],
            "tactic": ttp["tactic"],
            "technique": ttp["technique"],
            "apt_group": ttp["apt_group"],
            "tools_evaluated": security_tools,
            "tool_evaluations": tool_evaluations,
            "detected_by": detected_by,
            "overall_detection_rate": len(detected_by) / len(security_tools) if security_tools else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        self.evaluations.append(evaluation)
        self._save_evaluations()
        
        # Visualize detection results
        self._visualize_detection(evaluation)
        
        logging.info(f"Security evaluation for {ttp['technique']}: {len(detected_by)}/{len(security_tools)} tools detected it")
        return evaluation
    
    def evaluate_attack_chain(self, attack_chain):
        """Evaluate an entire attack chain against security tools"""
        chain_evaluations = []
        overall_detection = True
        
        # Get all unique tools targeted in the chain
        all_tools = set()
        for ttp in attack_chain:
            all_tools.update(ttp["tools_targeted"])
        
        security_tools = list(all_tools)
        
        for ttp in attack_chain:
            eval_result = self.evaluate_ttp(ttp, security_tools)
            chain_evaluations.append(eval_result)
            
            # Chain is successful if any step evades all detection
            if len(eval_result["detected_by"]) == 0:
                overall_detection = False
        
        chain_result = {
            "chain_length": len(attack_chain),
            "tactics": [ttp["tactic"] for ttp in attack_chain],
            "techniques": [ttp["technique"] for ttp in attack_chain],
            "apt_group": attack_chain[0]["apt_group"],
            "evaluations": chain_evaluations,
            "chain_detected": overall_detection,
            "timestamp": datetime.now().isoformat()
        }
        
        # Visualize chain evaluation
        self._visualize_chain_evaluation(chain_result)
        
        logging.info(f"Attack chain evaluation: {'Detected' if overall_detection else 'Not fully detected'}")
        return chain_result
    
    def _visualize_detection(self, evaluation):
        """Create visualization for tool detection"""
        plt.figure(figsize=(10, 6))
        
        tools = list(evaluation["tool_evaluations"].keys())
        detection_probs = [evaluation["tool_evaluations"][t]["detection_probability"] for t in tools]
        detected = [evaluation["tool_evaluations"][t]["detected"] for t in tools]
        
        # Create colormap based on detection
        colors = ['green' if d else 'red' for d in detected]
        
        plt.bar(tools, detection_probs, color=colors, alpha=0.7)
        plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Detection Threshold')
        
        plt.title(f"Security Tool Detection for: {evaluation['technique']}")
        plt.ylabel("Detection Probability")
        plt.xlabel("Security Tools")
        plt.ylim(0, 1)
        
        for i, v in enumerate(detection_probs):
            plt.text(i, v + 0.05, f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        filename = os.path.join(self.results_dir, f"detection_{evaluation['ttp_id']}.png")
        plt.savefig(filename)
        plt.close()
        
        logging.info(f"Detection visualization saved to {filename}")
        return filename
    
    def _visualize_chain_evaluation(self, chain_result):
        """Visualize attack chain detection"""
        plt.figure(figsize=(12, 8))
        
        # Create a step chart for the attack chain
        steps = len(chain_result["tactics"])
        x = np.arange(steps)
        
        # Calculate detection rates at each step
        detection_rates = [eval_result["overall_detection_rate"] for eval_result in chain_result["evaluations"]]
        
        plt.bar(x, detection_rates, alpha=0.7)
        plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Detection Threshold')
        
        plt.title(f"Attack Chain Detection: {chain_result['apt_group']}")
        plt.ylabel("Detection Rate")
        plt.xlabel("Attack Chain Step")
        plt.xticks(x, [f"{i+1}: {t}" for i, t in enumerate(chain_result["tactics"])], rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add technique labels
        for i, technique in enumerate(chain_result["techniques"]):
            plt.annotate(technique, (i, 0.05), rotation=90, fontsize=8, ha='center')
        
        plt.tight_layout()
        filename = os.path.join(self.results_dir, f"chain_{uuid.uuid4().hex[:8]}.png")
        plt.savefig(filename)
        plt.close()
        
        logging.info(f"Chain visualization saved to {filename}")
        return filename
    
    def get_tool_effectiveness(self, tool_name=None):
        """Get effectiveness statistics for security tools"""
        if not self.evaluations:
            return None
            
        tool_stats = {}
        
        for eval_result in self.evaluations:
            tools_evaluated = eval_result["tools_evaluated"]
            
            for tool in tools_evaluated:
                if tool_name and tool != tool_name:
                    continue
                    
                if tool not in tool_stats:
                    tool_stats[tool] = {
                        "evaluated_count": 0,
                        "detection_count": 0,
                        "detection_rate": 0,
                        "avg_detection_probability": 0,
                        "tactics_detected": {}
                    }
                
                tool_result = eval_result["tool_evaluations"].get(tool, {})
                detected = tool_result.get("detected", False)
                
                tool_stats[tool]["evaluated_count"] += 1
                if detected:
                    tool_stats[tool]["detection_count"] += 1
                    
                    # Track effectiveness by tactic
                    tactic = eval_result["tactic"]
                    if tactic not in tool_stats[tool]["tactics_detected"]:
                        tool_stats[tool]["tactics_detected"][tactic] = 0
                    tool_stats[tool]["tactics_detected"][tactic] += 1
                
                # Update avg detection probability
                current_avg = tool_stats[tool]["avg_detection_probability"]
                current_count = tool_stats[tool]["evaluated_count"]
                new_prob = tool_result.get("detection_probability", 0)
                tool_stats[tool]["avg_detection_probability"] = (current_avg * (current_count - 1) + new_prob) / current_count
        
        # Calculate detection rates
        for tool in tool_stats:
            if tool_stats[tool]["evaluated_count"] > 0:
                tool_stats[tool]["detection_rate"] = tool_stats[tool]["detection_count"] / tool_stats[tool]["evaluated_count"]
        
        # Return stats for specific tool or all tools
        if tool_name and tool_name in tool_stats:
            return tool_stats[tool_name]
        return tool_stats


class RedTeamAssistant:
    """Main class for the Red Team Assistant framework"""
    
    def __init__(self, config_file=None):
        """Initialize the Red Team Assistant"""
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.mitre = MITREIntegration()
        self.db = TTPDatabase(self.mitre)
        self.apt_profiler = APTProfiler(self.mitre)
        self.feature_extractor = FeatureExtractor(self.db)
        self.ml_model = RedTeamML()
        self.ml_model.set_feature_extractor(self.feature_extractor)
        self.security_evaluator = SecurityToolEvaluator()
        
        logging.info("Red Team Assistant initialized")
    
    def _load_config(self, config_file):
        """Load configuration from YAML file"""
        config = {
            "data_dir": "data",
            "models_dir": "models",
            "results_dir": "results",
            "use_mitre_data": True,
            "use_tensorflow": TENSORFLOW_AVAILABLE,
            "max_attack_chain_length": 5
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    config.update(loaded_config)
                logging.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logging.error(f"Error loading configuration: {e}")
        
        # Create necessary directories
        for dir_key in ['data_dir', 'models_dir', 'results_dir']:
            os.makedirs(config[dir_key], exist_ok=True)
        
        return config
    
    def add_ttp(self, tactic, technique, apt_group, skills_required=None, tools_targeted=None, mitre_id=None):
        """Add a new TTP to the database"""
        if skills_required is None:
            skills_required = ["computer_architecture", "networking"]
        if tools_targeted is None:
            tools_targeted = ["EDR", "SIEM"]
        if mitre_id is None:
            mitre_id = f"T{np.random.randint(1000, 9999)}"
        
        ttp_id = self.db.add_ttp(tactic, technique, apt_group, skills_required, tools_targeted, mitre_id)
        return ttp_id
    
    def generate_apt_ttp(self, tactic, apt_group=None):
        """Generate an APT-like TTP"""
        if apt_group is None:
            # Choose a random APT group
            apt_groups = self.apt_profiler.get_all_apt_names()
            if apt_groups:
                apt_group = np.random.choice(apt_groups)
            else:
                apt_group = "APT29"  # Default
        
        ttp_id = self.apt_profiler.generate_apt_ttp(self.db, tactic, apt_group)
        return ttp_id
    
    def train_models(self):
        """Train ML models with current data"""
        logging.info("Extracting features for model training...")
        X, y = self.feature_extractor.extract_features()
        
        if X is None or len(X) < 5:
            logging.warning("Not enough data for model training. Add more TTPs first.")
            return False
        
        # Train classifier
        classifier_accuracy = self.ml_model.train_classifier(X, y)
        
        # Generate attack chains for sequence model training
        attack_chains = []
        for _ in range(max(50, len(self.db.ttps) * 2)):
            if not self.db.ttps:
                break
                
            # Choose random starting TTP
            start_ttp = np.random.choice(self.db.ttps)
            chain = self.db.get_attack_chain(start_ttp["id"], max_length=self.config["max_attack_chain_length"])
            if len(chain) > 1:
                attack_chains.append(chain)
        
        # Train sequence model if we have enough chains
        sequence_accuracy = None
        if attack_chains and len(attack_chains) >= 10:
            logging.info(f"Training sequence model with {len(attack_chains)} attack chains")
            X_seq, y_seq = self.feature_extractor.extract_sequence_features(attack_chains)
            if X_seq is not None and len(X_seq) > 0:
                sequence_accuracy = self.ml_model.train_sequence_model(X_seq, y_seq)
        
        return {
            "classifier_accuracy": classifier_accuracy,
            "sequence_accuracy": sequence_accuracy,
            "ttps_used": len(self.db.ttps),
            "attack_chains_used": len(attack_chains) if attack_chains else 0
        }
    
    def generate_attack_chain(self, start_tactic=None, apt_group=None, length=3):
        """Generate a realistic attack chain"""
        # Choose a starting tactic if not specified
        if start_tactic is None:
            start_tactic = np.random.choice([
                "reconnaissance", "initial-access", "execution", 
                "persistence", "privilege-escalation", "defense-evasion"
            ])
        
        # Choose an APT group if not specified
        if apt_group is None:
            apt_groups = self.apt_profiler.get_all_apt_names()
            if apt_groups:
                apt_group = np.random.choice(apt_groups)
            else:
                apt_group = "APT29"  # Default
        
        logging.info(f"Generating attack chain: {start_tactic} / {apt_group}")
        
        # Generate first TTP if needed
        start_ttp_id = None
        existing_ttps = self.db.get_ttps_by_tactic(start_tactic)
        existing_ttps = [t for t in existing_ttps if t["apt_group"] == apt_group]
        
        if existing_ttps:
            start_ttp_id = np.random.choice(existing_ttps)["id"]
        else:
            start_ttp_id = self.generate_apt_ttp(start_tactic, apt_group)
        
        # Get attack chain
        chain = self.db.get_attack_chain(start_ttp_id, max_length=length)
        
        # If chain is too short, add more TTPs
        while len(chain) < length:
            # Find the last tactic in chain
            last_tactic = chain[-1]["tactic"] if chain else start_tactic
            
            # Predict next likely tactic
            next_tactic = self._predict_next_tactic(last_tactic)
            
            # Generate new TTP for this tactic
            new_ttp_id = self.generate_apt_ttp(next_tactic, apt_group)
            
            # Add to chain
            new_ttp = self.db.get_ttp(new_ttp_id)
            chain.append(new_ttp)
            
            # Add relationship to graph
            if len(chain) > 1:
                self.db.add_relationship(
                    chain[-2]["id"], 
                    chain[-1]["id"], 
                    "precedes", 
                    probability=0.7
                )
        
        logging.info(f"Generated attack chain with {len(chain)} steps")
        return chain
    
    def _predict_next_tactic(self, current_tactic):
        """Predict next likely tactic in attack chain"""
        # Simple transition model for tactics
        tactic_transitions = {
            "reconnaissance": ["initial-access", "reconnaissance"],
            "initial-access": ["execution", "persistence", "defense-evasion"],
            "execution": ["privilege-escalation", "defense-evasion", "credential-access"],
            "persistence": ["privilege-escalation", "defense-evasion", "discovery"],
            "privilege-escalation": ["defense-evasion", "credential-access", "discovery"],
            "defense-evasion": ["credential-access", "discovery", "lateral-movement"],
            "credential-access": ["discovery", "lateral-movement", "collection"],
            "discovery": ["lateral-movement", "collection", "command-and-control"],
            "lateral-movement": ["collection", "command-and-control", "exfiltration"],
            "collection": ["command-and-control", "exfiltration", "impact"],
            "command-and-control": ["exfiltration", "impact"],
            "exfiltration": ["impact"],
            "impact": ["exfiltration", "impact"]
        }
        
        if current_tactic in tactic_transitions:
            return np.random.choice(tactic_transitions[current_tactic])
        else:
            # Default tactics if unknown
            return np.random.choice([
                "execution", "persistence", "privilege-escalation", 
                "defense-evasion", "discovery", "lateral-movement"
            ])
    
    def evaluate_ttp(self, ttp_id):
        """Evaluate a TTP against security tools"""
        ttp = self.db.get_ttp(ttp_id)
        if not ttp:
            logging.error(f"TTP not found: {ttp_id}")
            return None
        
        evaluation = self.security_evaluator.evaluate_ttp(ttp)
        return evaluation
    
    def evaluate_attack_chain(self, chain):
        """Evaluate an attack chain against security tools"""
        if not chain or len(chain) == 0:
            logging.error("Empty attack chain provided")
            return None
        
        evaluation = self.security_evaluator.evaluate_attack_chain(chain)
        return evaluation
    
    def visualize_database(self):
        """Visualize the TTP database"""
        return self.db.visualize_graph()
    
    def export_report(self, output_file="red_team_report.json"):
        """Export a comprehensive report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "ttps_count": len(self.db.ttps),
            "apt_groups": len(self.apt_profiler.apt_profiles),
            "security_tools_evaluated": list(self.security_evaluator.get_tool_effectiveness().keys())
                                   if self.security_evaluator.get_tool_effectiveness() else [],
            "model_status": {
                "classifier_trained": self.ml_model.is_classifier_trained,
                "sequence_model_trained": self.ml_model.is_sequence_trained
            },
            "recommendations": self._generate_recommendations()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Report exported to {output_file}")
        return report
    
    def _generate_recommendations(self):
        """Generate security recommendations based on evaluations"""
        recommendations = []
        
        # Get tool effectiveness data
        tool_stats = self.security_evaluator.get_tool_effectiveness()
        if not tool_stats:
            return recommendations
        
        # Identify weak security tools
        weak_tools = []
        for tool, stats in tool_stats.items():
            if stats["detection_rate"] < 0.6 and stats["evaluated_count"] >= 5:
                weak_tools.append({
                    "tool": tool,
                    "detection_rate": stats["detection_rate"],
                    "evaluated_count": stats["evaluated_count"]
                })
        
        if weak_tools:
            recommendations.append({
                "title": "Improve Security Tool Coverage",
                "description": "The following security tools have low detection rates and may need configuration improvements:",
                "details": weak_tools
            })
        
        # Identify tactics with low detection
        tactic_detection = {}
        for eval_item in self.security_evaluator.evaluations:
            tactic = eval_item["tactic"]
            detected = len(eval_item["detected_by"]) > 0
            
            if tactic not in tactic_detection:
                tactic_detection[tactic] = {"count": 0, "detected": 0}
            
            tactic_detection[tactic]["count"] += 1
            if detected:
                tactic_detection[tactic]["detected"] += 1
        
        # Calculate detection rates by tactic
        weak_tactics = []
        for tactic, stats in tactic_detection.items():
            if stats["count"] >= 3:  # Only consider tactics with enough samples
                detection_rate = stats["detected"] / stats["count"]
                if detection_rate < 0.5:
                    weak_tactics.append({
                        "tactic": tactic,
                        "detection_rate": detection_rate,
                        "count": stats["count"]
                    })
        
        if weak_tactics:
            recommendations.append({
                "title": "Address Tactical Blind Spots",
                "description": "The following tactics have low detection rates across your security tools:",
                "details": weak_tactics
            })
        
        return recommendations


def cli():
    """Command-line interface for Red Team Assistant"""
    parser = argparse.ArgumentParser(description="Red Team Assistant - APT Simulation Framework")
    
    # Configuration
    parser.add_argument("--config", help="Path to configuration file", default=None)
    
    # Commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add TTP command
    add_parser = subparsers.add_parser("add-ttp", help="Add a new TTP")
    add_parser.add_argument("--tactic", required=True, help="MITRE ATT&CK tactic")
    add_parser.add_argument("--technique", required=True, help="Technique name")
    add_parser.add_argument("--apt", required=True, help="APT group name")
    add_parser.add_argument("--skills", nargs="+", help="Required skills")
    add_parser.add_argument("--tools", nargs="+", help="Targeted security tools")
    add_parser.add_argument("--mitre-id", help="MITRE ATT&CK ID")
    
    # Generate TTP command
    gen_parser = subparsers.add_parser("generate-ttp", help="Generate an APT-like TTP")
    gen_parser.add_argument("--tactic", required=True, help="MITRE ATT&CK tactic")
    gen_parser.add_argument("--apt", help="APT group name")
    
    # Generate attack chain command
    chain_parser = subparsers.add_parser("generate-chain", help="Generate an attack chain")
    chain_parser.add_argument("--tactic", help="Starting tactic")
    chain_parser.add_argument("--apt", help="APT group name")
    chain_parser.add_argument("--length", type=int, default=3, help="Chain length")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate TTP or attack chain")
    eval_parser.add_argument("--ttp-id", help="TTP ID to evaluate")
    eval_parser.add_argument("--chain", action="store_true", help="Generate and evaluate an attack chain")
    
    # Train models command
    train_parser = subparsers.add_parser("train", help="Train ML models")
    
    # Visualization commands
    vis_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    vis_parser.add_argument("--type", choices=["graph", "stats"], default="graph", help="Visualization type")
    
    # Export report command
    report_parser = subparsers.add_parser("report", help="Export red team report")
    report_parser.add_argument("--output", default="red_team_report.json", help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize the assistant
    assistant = RedTeamAssistant(args.config)
    
    # Execute commands
    if args.command == "add-ttp":
        ttp_id = assistant.add_ttp(
            args.tactic, 
            args.technique, 
            args.apt,
            args.skills,
            args.tools,
            args.mitre_id
        )
        print(f"Added TTP with ID: {ttp_id}")
    
    elif args.command == "generate-ttp":
        ttp_id = assistant.generate_apt_ttp(args.tactic, args.apt)
        ttp = assistant.db.get_ttp(ttp_id)
        print(f"Generated TTP:")
        print(json.dumps(ttp, indent=2))
    
    elif args.command == "generate-chain":
        chain = assistant.generate_attack_chain(args.tactic, args.apt, args.length)
        print(f"Generated attack chain with {len(chain)} steps:")
        for i, ttp in enumerate(chain):
            print(f"{i+1}. {ttp['tactic']}: {ttp['technique']}")
        
        # Ask if user wants to evaluate the chain
        if input("Evaluate this chain against security tools? (y/n): ").lower() == 'y':
            evaluation = assistant.evaluate_attack_chain(chain)
            print(f"Chain {'fully detected' if evaluation['chain_detected'] else 'not fully detected'}")
    
    elif args.command == "evaluate":
        if args.ttp_id:
            evaluation = assistant.evaluate_ttp(args.ttp_id)
            print(f"TTP evaluated: detected by {len(evaluation['detected_by'])}/{len(evaluation['tools_evaluated'])} tools")
        elif args.chain:
            chain = assistant.generate_attack_chain()
            evaluation = assistant.evaluate_attack_chain(chain)
            print(f"Generated chain {'fully detected' if evaluation['chain_detected'] else 'not fully detected'}")
        else:
            print("Please specify --ttp-id or --chain")
    
    elif args.command == "train":
        results = assistant.train_models()
        print("Model training complete:")
        print(f"Classifier accuracy: {results['classifier_accuracy']:.2f}")
        if results['sequence_accuracy']:
            print(f"Sequence model accuracy: {results['sequence_accuracy']:.2f}")
        else:
            print("Sequence model not trained (insufficient data or TensorFlow not available)")
    
    elif args.command == "visualize":
        if args.type == "graph":
            graph_file = assistant.visualize_database()
            print(f"Graph visualization saved to {graph_file}")
        else:
            # Generate statistics visualization
            tool_stats = assistant.security_evaluator.get_tool_effectiveness()
            if tool_stats:
                plt.figure(figsize=(10, 6))
                tools = list(tool_stats.keys())
                rates = [stats["detection_rate"] for stats in tool_stats.values()]
                
                plt.bar(tools, rates)
                plt.title("Security Tool Detection Rates")
                plt.ylabel("Detection Rate")
                plt.xlabel("Security Tools")
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, 1)
                plt.tight_layout()
                
                stats_file = os.path.join(assistant.config["results_dir"], "tool_stats.png")
                plt.savefig(stats_file)
                plt.close()
                print(f"Statistics visualization saved to {stats_file}")
            else:
                print("No evaluation data available for statistics")
    
    elif args.command == "report":
        report = assistant.export_report(args.output)
        print(f"Report exported to {args.output}")
        
        # Print key findings
        if "recommendations" in report and report["recommendations"]:
            print("\nKey recommendations:")
            for rec in report["recommendations"]:
                print(f"- {rec['title']}")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logging.error(f"Error: {e}")
        print(f"Error: {e}")
        sys.exit(1)
