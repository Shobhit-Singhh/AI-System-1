from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from typing import List, Tuple
from langchain_core.documents import Document

def LLM_instruction_manual_enricher(chunks: List[Document]) -> List[Tuple[str, str]]:
    """
    Enrich and expand existing condition-behavior chunks into comprehensive instruction manuals.
    
    Args:
        chunks: List of Document objects containing existing condition-behavior descriptions
        
    Returns:
        List of tuples where each tuple contains (enriched_condition, enriched_instruction_manual)
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)  # Balanced creativity and consistency
    
    # Define the output structure
    class EnrichedInstructionManual(BaseModel):
        enriched_condition: str = Field(description="Detailed, comprehensive condition description with context and triggers")
        enriched_instruction_manual: str = Field(description="Complete, detailed instruction manual for AI agent behavior")
    
    # Define the prompt template
    template = PromptTemplate(
        input_variables=["chunk_content"],
        template="""You are an expert AI instruction manual writer and behavioral specification designer.

        Your task is to take an existing condition-behavior description and transform it into a comprehensive, detailed instruction manual that an AI agent can follow precisely.

        **Input Content (Existing Condition-Behavior Description):**
        {chunk_content}

        **Your Mission:**
        Analyze the provided chunk which contains a condition and corresponding AI behavior description. Your goal is to ENRICH and EXPAND both the condition and the behavioral instructions into a comprehensive manual.

        **Enrichment Framework:**

        1. **Condition Enrichment:**
        - Expand the condition with detailed context, scenarios, and triggers
        - Add specific indicators, keywords, or patterns that would activate this condition
        - Include edge cases and variations of the condition
        - Specify when this condition applies vs when it doesn't
        - Add measurable criteria for condition detection

        2. **Instruction Manual Enrichment:**
        - Transform basic behavior descriptions into step-by-step detailed instructions
        - Add specific response formats, tone guidelines, and communication styles
        - Include examples of appropriate responses and outputs
        - Specify what to avoid and common pitfalls
        - Add escalation procedures and fallback behaviors
        - Include quality checks and success criteria
        - Provide specific phrases, templates, or frameworks to use

        **Quality Standards:**
        - Make instructions so detailed that any AI agent could follow them consistently
        - Include specific examples and counter-examples
        - Add measurable success criteria
        - Ensure completeness - cover all aspects of the desired behavior
        - Make conditions precise enough to avoid ambiguity
        - Include context awareness and situational adaptability

        **Example Transformation:**
        Input: "When user asks about medical advice, be cautious and refer to professionals"
        Output Condition: "User query contains medical symptoms, health concerns, diagnostic requests, treatment questions, medication inquiries, or requests for medical opinions. Indicators include: pain descriptions, symptom lists, 'should I see a doctor', medication names, health condition names, diagnostic terminology."
        Output Manual: "1. Acknowledge the user's concern with empathy. 2. Clearly state 'I cannot provide medical diagnosis or treatment advice.' 3. Recommend consulting healthcare professionals. 4. Offer to help find general health information or direct to reputable medical resources. 5. Avoid any language that could be interpreted as diagnostic. 6. Never suggest specific medications or treatments. 7. If emergency indicators present (chest pain, difficulty breathing, etc.), immediately recommend emergency services. 8. Maintain supportive tone while being firm about limitations."

        **Output Format:**
        {format_instructions}

        **Remember:** Transform the basic condition-behavior into a comprehensive manual that leaves no room for interpretation or inconsistent application.""",
        validate_template=True,
        partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=EnrichedInstructionManual).get_format_instructions()}
    )
    
    # Build the chain
    output_parser = PydanticOutputParser(pydantic_object=EnrichedInstructionManual)
    chain = template | llm | output_parser
    
    enriched_instruction_manuals = []
    
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, Document):
            raise ValueError(f"Chunk {i} must be an instance of Document.")
        elif not chunk.page_content.strip():
            print(f"Warning: Chunk {i} has empty content, skipping...")
            continue
        
        try:
            print(f"Enriching instruction manual for chunk {i+1}/{len(chunks)}...")
            
            # Generate enriched instruction manual for this chunk
            response = chain.invoke({
                "chunk_content": chunk.page_content
            })
            
            # Extract the enriched condition and instruction manual
            enriched_condition = response.enriched_condition
            enriched_manual = response.enriched_instruction_manual
            pair = (enriched_condition, enriched_manual)
            
            print(f"Generated enriched instruction manual from chunk {i+1}")
            print(f"  Enriched Condition: {enriched_condition}")
            print(f"  Manual Length: {len(enriched_manual)} characters")
            print()
            
            # Add to master list
            enriched_instruction_manuals.append(pair)
                
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            continue
    
    print(f"\nTotal enriched instruction manuals generated: {len(enriched_instruction_manuals)}")
    return enriched_instruction_manuals

def save_enriched_instruction_manuals(enriched_manuals: List[Tuple[str, str]], output_file: str = "enriched_instruction_manuals.txt"):
    """
    Save enriched instruction manuals to a file.
    
    Args:
        enriched_manuals: List of (enriched_condition, enriched_instruction_manual) tuples
        output_file: Path to output file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ENRICHED AI AGENT INSTRUCTION MANUALS\n")
            f.write("=" * 80 + "\n\n")
            
            for i, (condition, manual) in enumerate(enriched_manuals):
                f.write(f"MANUAL #{i+1}:\n")
                f.write("=" * 40 + "\n")
                f.write(f"CONDITION:\n{condition}\n\n")
                f.write(f"INSTRUCTION MANUAL:\n{manual}\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"Enriched instruction manuals saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving manuals to file: {e}")

def convert_manuals_to_rag_documents(enriched_manuals: List[Tuple[str, str]]) -> List[Document]:
    """
    Convert enriched instruction manuals to Document objects for RAG indexing.
    
    Args:
        enriched_manuals: List of (enriched_condition, enriched_instruction_manual) tuples
        
    Returns:
        List of Document objects where page_content contains the enriched condition 
        and metadata contains the corresponding enriched instruction manual
    """
    rag_documents = []
    
    for i, (condition, manual) in enumerate(enriched_manuals):
        doc = Document(
            page_content=condition,  # This will be indexed for similarity search
            metadata={
                "instruction_manual": manual,  # This will be retrieved as the detailed prompt instruction
                "manual_id": i,
                "type": "enriched_instruction_manual",
                "manual_length": len(manual)
            }
        )
        rag_documents.append(doc)
    
    return rag_documents

def example_usage():
    """Example of how to use the instruction manual enricher."""
    
    # Sample chunks with existing condition-behavior descriptions
    sample_chunks = [
        Document(page_content="""
        When a user asks about medical advice, the AI should be cautious and refer them to healthcare professionals.
        The agent should not provide diagnostic information or treatment recommendations.
        """),
        
        Document(page_content="""
        If the user seems frustrated or angry, the AI agent should remain calm, acknowledge their feelings,
        and try to de-escalate the situation by being empathetic and solution-focused.
        """),
        
        Document(page_content="""
        When handling sensitive personal information, the AI must prioritize privacy and security.
        It should not store, share, or reference personal data in future conversations.
        The agent should remind users about data privacy practices when appropriate.
        """),
        
        Document(page_content="""
        For technical coding questions, the AI should provide clear, working examples with explanations.
        Code should be well-commented and include error handling where appropriate.
        The agent should also suggest best practices and potential improvements.
        """)
    ]
    
    print("Starting instruction manual enrichment...")
    
    # Generate enriched manuals
    enriched_manuals = LLM_instruction_manual_enricher(sample_chunks)
    
    # Save to file
    save_enriched_instruction_manuals(enriched_manuals, "example_enriched_manuals.txt")
    
    # Convert to RAG documents
    rag_docs = convert_manuals_to_rag_documents(enriched_manuals)
    
    print(f"\nCreated {len(rag_docs)} RAG documents for indexing")
    print("\nExample enriched manual:")
    if rag_docs:
        print(f"Enriched Condition: {rag_docs[0].page_content}")
        print(f"Manual Preview: {rag_docs[0].metadata['instruction_manual']}")
    
    return enriched_manuals, rag_docs

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
    enriched_manuals, rag_docs = example_usage()