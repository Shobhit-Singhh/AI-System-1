from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from typing import List, Tuple
from langchain_core.documents import Document

def LLM_condition_instruction_generator(chunks: List[Document]) -> List[Tuple[str, str]]:
    """
    Generate condition-instruction pairs from document chunks for dynamic RAG prompting.
    
    Args:
        chunks: List of Document objects containing content to analyze
        
    Returns:
        List of tuples where each tuple contains (condition, instruction)
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)  # Balanced creativity and consistency
    
    # Define the output structure
    class ConditionInstructionPairs(BaseModel):
        pairs: List[Tuple[str, str]] = Field(
            description="List of condition-instruction pairs where each tuple contains (condition, instruction)"
        )
    
    # Define the prompt template
    template = PromptTemplate(
        input_variables=["chunk_content"],
        template="""You are an expert prompt engineer and behavioral instruction designer.

Your task is to analyze the provided document chunk and generate comprehensive condition-instruction pairs that can be used in a dynamic RAG (Retrieval-Augmented Generation) system.

**Input Content:**
{chunk_content}

**Analysis Framework:**
1. **Content Analysis**: Deeply analyze the chunk to understand its core concepts, themes, and knowledge domains
2. **Context Recognition**: Identify different scenarios, situations, or contexts where this content would be relevant
3. **Behavioral Mapping**: Determine what specific behaviors, responses, or approaches the LLM should adopt for each context
4. **Instruction Crafting**: Create precise, actionable instructions that guide LLM behavior for each identified condition

**Generation Guidelines:**
- **Conditions** should be specific, measurable scenarios that can be matched against user queries or contexts
- **Instructions** should be clear, actionable behavioral directives that tell the LLM exactly how to respond
- Generate multiple diverse condition-instruction pairs per chunk (aim for 3-5 pairs minimum)
- Cover different aspects: technical depth, audience level, response style, formatting, tone, etc.
- Make conditions specific enough to avoid overlap but broad enough to be useful
- Ensure instructions are practical and implementable

**Examples of Good Pairs:**
- Condition: "User asks about technical implementation details of quantum computing algorithms"
  Instruction: "Provide step-by-step technical explanations with mathematical formulations, code examples where applicable, and mention specific quantum gates and circuit designs"

- Condition: "User needs beginner-friendly explanation of complex medical terminology"
  Instruction: "Use simple analogies, avoid jargon, provide everyday examples, and break down complex terms into understandable components"

**Output Format:**
{format_instructions}

**Important:** Each condition should be a distinct scenario, and each instruction should provide specific behavioral guidance for that scenario.""",
        validate_template=True,
        partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=ConditionInstructionPairs).get_format_instructions()}
    )
    
    # Build the chain
    output_parser = PydanticOutputParser(pydantic_object=ConditionInstructionPairs)
    chain = template | llm | output_parser
    
    all_condition_instruction_pairs = []
    
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, Document):
            raise ValueError(f"Chunk {i} must be an instance of Document.")
        elif not chunk.page_content.strip():
            print(f"Warning: Chunk {i} has empty content, skipping...")
            continue
        
        try:
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # Generate condition-instruction pairs for this chunk
            response = chain.invoke({
                "chunk_content": chunk.page_content
            })
            
            # Extract pairs from response
            pairs = response.pairs
            
            print(f"Generated {len(pairs)} condition-instruction pairs from chunk {i+1}")
            
            # Add to master list
            all_condition_instruction_pairs.extend(pairs)
            
            # Optional: Print pairs for review
            for j, (condition, instruction) in enumerate(pairs):
                print(f"  Pair {j+1}:")
                print(f"    Condition: {condition[:100]}...")
                print(f"    Instruction: {instruction[:100]}...")
                print()
                
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            continue
    
    print(f"\nTotal condition-instruction pairs generated: {len(all_condition_instruction_pairs)}")
    return all_condition_instruction_pairs

def save_condition_instruction_pairs(pairs: List[Tuple[str, str]], output_file: str = "condition_instruction_pairs.txt"):
    """
    Save condition-instruction pairs to a file in a format suitable for RAG indexing.
    
    Args:
        pairs: List of (condition, instruction) tuples
        output_file: Path to output file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (condition, instruction) in enumerate(pairs):
                f.write(f"PAIR_{i+1}:\n")
                f.write(f"CONDITION: {condition}\n")
                f.write(f"INSTRUCTION: {instruction}\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"Condition-instruction pairs saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving pairs to file: {e}")

def convert_pairs_to_rag_documents(pairs: List[Tuple[str, str]]) -> List[Document]:
    """
    Convert condition-instruction pairs to Document objects for RAG indexing.
    
    Args:
        pairs: List of (condition, instruction) tuples
        
    Returns:
        List of Document objects where page_content contains the condition 
        and metadata contains the corresponding instruction
    """
    rag_documents = []
    
    for i, (condition, instruction) in enumerate(pairs):
        doc = Document(
            page_content=condition,  # This will be indexed for similarity search
            metadata={
                "instruction": instruction,  # This will be retrieved as the prompt instruction
                "pair_id": i,
                "type": "condition_instruction_pair"
            }
        )
        rag_documents.append(doc)
    
    return rag_documents

def example_usage():
    """Example of how to use the condition-instruction generator."""
    
    # Sample chunks (replace with your actual chunks)
    sample_chunks = [
        Document(page_content="""
        Machine learning algorithms require careful hyperparameter tuning to achieve optimal performance. 
        Common hyperparameters include learning rate, batch size, number of epochs, and regularization strength. 
        Grid search and random search are traditional approaches, while more advanced methods like Bayesian 
        optimization and evolutionary algorithms can be more efficient for complex parameter spaces.
        """),
        
        Document(page_content="""
        Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement 
        to process information. Quantum bits (qubits) can exist in multiple states simultaneously, 
        allowing quantum computers to explore many solutions in parallel. However, quantum systems 
        are fragile and require extremely low temperatures and isolation from environmental interference.
        """),
        
        Document(page_content="""
        In clinical psychology, cognitive behavioral therapy (CBT) is an evidence-based approach that 
        focuses on identifying and changing negative thought patterns and behaviors. CBT typically 
        involves homework assignments, thought records, and behavioral experiments. The therapeutic 
        relationship and patient engagement are crucial factors for successful outcomes.
        """)
    ]
    
    print("Starting condition-instruction pair generation...")
    
    # Generate pairs
    pairs = LLM_condition_instruction_generator(sample_chunks)
    
    # Save to file
    save_condition_instruction_pairs(pairs, "example_pairs.txt")
    
    # Convert to RAG documents
    rag_docs = convert_pairs_to_rag_documents(pairs)
    
    print(f"\nCreated {len(rag_docs)} RAG documents for indexing")
    print("\nExample RAG document:")
    if rag_docs:
        print(f"Content (for indexing): {rag_docs[0].page_content}")
        print(f"Instruction (for prompting): {rag_docs[0].metadata['instruction']}")
    
    return pairs, rag_docs

# Advanced function for topic-aware pair generation
def LLM_topic_aware_condition_instruction_generator(classified_chunks: dict[str, List[Document]]) -> dict[str, List[Tuple[str, str]]]:
    """
    Generate condition-instruction pairs with topic awareness.
    
    Args:
        classified_chunks: Dictionary mapping topic names to lists of Document objects
        
    Returns:
        Dictionary mapping topic names to lists of condition-instruction pairs
    """
    topic_pairs = {}
    
    for topic, chunks in classified_chunks.items():
        print(f"\nProcessing topic: {topic}")
        pairs = LLM_condition_instruction_generator(chunks)
        topic_pairs[topic] = pairs
        
        # Add topic-specific context to conditions
        enhanced_pairs = []
        for condition, instruction in pairs:
            enhanced_condition = f"[Topic: {topic}] {condition}"
            enhanced_pairs.append((enhanced_condition, instruction))
        
        topic_pairs[topic] = enhanced_pairs
    
    return topic_pairs

if __name__ == "__main__":
    # Run the example
    pairs, rag_docs = example_usage()