from langchain_community.document_loaders import TextLoader

data = TextLoader("chatrapati_shivaji_raje.txt") 
#print(data)

docs = data.load()

#this TextLoader just take your .txt file and make a data object

# that data object is like , [(your first docs),(your second docs),(your third docs)] ,

# but note that your first doc , your second docs and your third docs has two fields which is :

# meta_data and file page_content means actually it like ,

# [(meta_data={},page_content={}),(meta_data={},page_content={}),(meta_data={},page_content={})]

#but currently we have a [(chatrapati_shivaji_raje.txt)] , but actuallu it like :

# [(meta_data={},page_content={here the actual text which is in the chatrapati_shivaji_raje.txt }))]

#print(docs)
print("================================")
print(docs[0].page_content)