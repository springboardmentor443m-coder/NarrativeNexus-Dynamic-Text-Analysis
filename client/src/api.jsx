import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000", // FastAPI backend
});

export const analyzeFile = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await API.post("/analyze", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data; // The JSON response from FastAPI
};
