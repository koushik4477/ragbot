from rag_system import PersonaRAGSystem  # import your class
import json

# Initialize for the persona you want to inspect
persona_id = 1
rag = PersonaRAGSystem(persona_id)

# Retrieve all knowledge stored for that persona
knowledge_items = rag.get_all_knowledge()

# Print in readable format
if not knowledge_items:
    print(f"No knowledge found for persona {persona_id}.")
else:
    print(f"✅ Found {len(knowledge_items)} knowledge entries for persona {persona_id}:\n")
    for item in knowledge_items:
        print(f"ID: {item['id']}")
        print(f"Category: {item['metadata'].get('category', 'unknown')}")
        print(f"Text: {item['text']}\n{'-'*70}")

# Optionally save to a JSON file
with open(f"persona_{persona_id}_knowledge_dump.json", "w", encoding="utf-8") as f:
    json.dump(knowledge_items, f, indent=4, ensure_ascii=False)
    print(f"\n📁 Exported all knowledge to persona_{persona_id}_knowledge_dump.json")
