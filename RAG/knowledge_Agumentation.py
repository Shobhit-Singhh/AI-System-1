from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from typing import Tuple
from langchain_core.documents import Document

def LLM_classification_chain(chunks: list[Document]) -> dict[str, list[Document]]:
    """Classify document chunks into topics using LLM."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Define the output parser using Pydantic
    class Prompt_output(BaseModel):
        current_topic_summary: Tuple[str, str] = Field(
            description="Topic of the chunk, and the summary of the topic"
        )
    
    # Define the prompt template
    template = PromptTemplate(
        input_variables=["topic_name_description", "chunk"],
        template="""You are an expert prompt engineer and topic detector.
Your task is to classify a provided document chunk into an existing topic or identify it as a new topic, based on provided topic descriptions.

You are given:
- **topic_name_description**: {topic_name_description}
- **chunk**: {chunk}

### Instructions:
1. If the chunk fits an existing topic, provide the topic name and an updated summary that adds the new information.
2. If the chunk introduces a new topic, create a topic name and summary.

### Output Format:
{format_instructions}""",
        validate_template=True,
        partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=Prompt_output).get_format_instructions()}
    )
    
    # Build the chain using modern syntax
    output_parser = PydanticOutputParser(pydantic_object=Prompt_output)
    chain = template | llm | output_parser
    
    topic_name_description = {}
    classified_chunks = {}
    
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, Document):
            raise ValueError(f"Chunk {i} must be an instance of Document.")
        elif not chunk.page_content:
            raise ValueError(f"Chunk {i} must have non-empty page content.")
        
        try:
            # Invoke the chain with the chunk and topic name description
            response = chain.invoke({
                "topic_name_description": str(topic_name_description) if topic_name_description else "No existing topics",
                "chunk": chunk.page_content,
            })
            
            # Extract topic name and summary from response
            topic_name, summary = response.current_topic_summary
            
            # Update the topic name description with the response
            topic_name_description[topic_name] = summary
            
            # Add chunk to classified chunks (fixed the append logic)
            if topic_name in classified_chunks:
                classified_chunks[topic_name].append(chunk)
            else:
                classified_chunks[topic_name] = [chunk]
                
            print(f"Chunk {i+1} classified as: {topic_name}")
            
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            # Assign to a default topic if classification fails
            default_topic = "Unclassified"
            if default_topic in classified_chunks:
                classified_chunks[default_topic].append(chunk)
            else:
                classified_chunks[default_topic] = [chunk]
    
    return classified_chunks

def LLM_content_enhancement_chain(chunks: dict[str, list[Document]]) -> dict[str, str]:
    """Enhance and paraphrase chunks to create RAG-friendly content."""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)  # Slightly higher temperature for creativity
    
    # Define the prompt template for content enhancement
    template = PromptTemplate(
        input_variables=["chunk"],
        template="""You are an expert content enhancer and paraphraser specializing in creating RAG-optimized content.

Your task is to transform the provided document chunk into enriched, elaborated content that is highly suitable for Retrieval-Augmented Generation (RAG) systems.

You are given:
- **chunk**: {chunk}

### Instructions:
1. **Preserve all key information** from the original content - do not lose any important facts, concepts, or details
2. **Expand and elaborate** on concepts mentioned in the chunk by adding relevant context and explanations
3. **Improve clarity and readability** by restructuring sentences and using clearer language
4. **Add semantic richness** by including related terminology, synonyms, and conceptual connections
5. **Ensure question-answer compatibility** - structure the content so it can effectively answer various types of questions
6. **Maintain factual accuracy** - do not introduce information not supported by the original content
7. **Create comprehensive coverage** - ensure the enhanced content covers all aspects mentioned in the original

### Output Format:
Provide the enhanced, elaborated, and paraphrased content that maintains all original information while being more comprehensive and RAG-friendly. The output should be substantially more detailed and informative than the input while remaining accurate.""",
        validate_template=True,
    )
    
    # Build the chain using modern syntax
    chain = template | llm
    
    enhanced_content = {}
    
    for topic, chunk_list in chunks.items():
        if not isinstance(chunk_list, list) or not all(isinstance(c, Document) for c in chunk_list):
            raise ValueError(f"Chunks for topic '{topic}' must be a list of Document instances.")
        
        # Concatenate the page content of all chunks for the topic
        concatenated_chunks = " ".join([c.page_content for c in chunk_list])
        
        try:
            # Invoke the chain with the concatenated chunks
            response = chain.invoke({"chunk": concatenated_chunks})
            
            # Extract the enhanced content from response
            if hasattr(response, 'content'):
                enhanced_text = response.content
            elif isinstance(response, str):
                enhanced_text = response
            else:
                enhanced_text = str(response)
            
            enhanced_content[topic] = enhanced_text
            print(f"Enhanced content for topic '{topic}' (Original: {len(concatenated_chunks)} chars → Enhanced: {len(enhanced_text)} chars)")
            
        except Exception as e:
            print(f"Error enhancing content for topic '{topic}': {e}")
            enhanced_content[topic] = concatenated_chunks  # Fallback to original content
    
    return enhanced_content

# Example usage function
def example_usage():
    """Example of how to use the functions."""
    # Create sample documents for testing
    sample_chunks = [
        Document(page_content="Quantum computing uses quantum mechanical phenomena to process information in ways that classical computers cannot."),
        Document(page_content="Machine learning algorithms can identify patterns in large datasets and make predictions based on historical data."),
        Document(page_content="Quantum entanglement is a physical phenomenon that occurs when particles become interconnected and the state of each particle cannot be described independently."),
        Document(page_content="Deep learning models use neural networks with multiple layers to learn complex patterns from data."),
        Document(page_content="Quantum supremacy refers to the point where quantum computers can perform calculations that are practically impossible for classical computers.")
    ]
    
    print("Number of chunks created:", len(sample_chunks))
    
    # Classify chunks
    classified_chunks = LLM_classification_chain(sample_chunks)
    print("\nClassified Chunks:")
    for topic, chunks in classified_chunks.items():
        print(f"  {topic}: {len(chunks)} chunks")
    
    # Enhance content quality
    print("\nEnhancing content for RAG optimization...")
    enhanced_content = LLM_content_enhancement_chain(classified_chunks)
    
    return classified_chunks, enhanced_content

if __name__ == "__main__":
    # Run example
    # example_usage()
    
    from Loader import load_text
    from Splitter import split_texts
    
    file_path = "./../data/Documents/DSM5.txt"
    document = load_text(file_path)[0].page_content
    chunks = split_texts(document, chunk_size=1000, chunk_overlap=300)
    
    classified_chunks = LLM_classification_chain(chunks)
    enhanced_content = LLM_content_enhancement_chain(classified_chunks)
    
    for topic, content in enhanced_content.items():
        print(f"\nTopic: {topic}\nContent:\n{content} \n")