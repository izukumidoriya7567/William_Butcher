import dotenv from "dotenv";
dotenv.config();
import {QdrantClient} from "@qdrant/js-client-rest";
import {HuggingFaceInferenceEmbeddings} from "@langchain/community/embeddings/hf";
import {ChatPromptTemplate} from "@langchain/core/prompts";
import {ChatGroq} from "@langchain/groq";
import {z} from "zod";
import express from "express";
import cors from "cors";
const app=express();
app.use(cors({
  origin: 'http://localhost:5173', // Allow only your frontend origin
  // For multiple origins: origin: ['http://localhost:5173', 'https://yourproductionapp.com'],
  // Or allow any origin: origin: '*', (less secure)
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  credentials: true // If you need to send cookies or auth headers
}));
const embeddings=new HuggingFaceInferenceEmbeddings({
    model: "sentence-transformers/all-MiniLM-L6-v2",
    apiKey: process.env.EMBEDDINGS_APIKEY,
});

const client=new QdrantClient({
    url:"https://d2abd0c8-b572-452f-9670-c776a382be87.us-east4-0.gcp.cloud.qdrant.io",
    apiKey:process.env.QDRANT_APIKEY,
});

const llm=new ChatGroq({
    model:"llama-3.3-70b-versatile",
    temperature:0.0,
    apiKey:process.env.LLM_APIKEY,
});

const answer = z.object({
  imp_words: z.string().describe("List the important legal terms or keywords that a Judiciary aspirant should remember from this context."),
  description: z.string().describe("Provide a descriptive explanation based on the given context, relevant to the question asked."),
  exact: z.string().describe("Quote the exact text from the context that directly answers the question.")
});

const structuredLlm=llm.withStructuredOutput(answer);

const generate=async(query,act)=>{
    try{
        const queryVector=await embeddings.embedQuery(query);
        const searchResult=await client.search(`${act}`,{
            vector:queryVector,
            limit:3,
        });
        const pay=searchResult.map((result)=>["system", result.payload.text]);
        const prompt=ChatPromptTemplate.fromMessages([
            ["system",`You are an AI assistant which is designed
                 to help my sister Anika Tripathi who is a Law Student and wants to prepare for Judiciary exams in India.
                 You will be provided with context, and you need to answer accordingly in a structured way, the structure is already provided to you.
                 So, you just need to answer the questions based on the context, structure and the question provided to you. 
                 If the question provided is out of context, please ask the user to put relevant questions in the Box.`],
            ...pay,
            ["human","{input}"],
        ]);
        const chain=prompt.pipe(structuredLlm);
        const response=await chain.invoke({input:`${query}`});
        return {type:"success",content:response};
    }
    catch(e){
        return {type:"error",content:e};
    }
}

app.get("/",(req,res)=>{
    res.status(200).send("Do you think you have got the bullocks: William Butcher");
})

app.get("/:act/:query",async(req,res)=>{
      const act=req.params.act;
      const query=req.params.query;
      const answer=await generate(query,act);
      console.log(answer);
      if(answer.type==="success"){
        res.status(200).json({type:"success",content:answer.content});
      }
      else{
        res.status(500).json({type:"error",content:answer.content});
      }
})
export default app;