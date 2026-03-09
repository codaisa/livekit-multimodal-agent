import express from "express";

const app = express();

app.post("/voice", (req, res) => {
  res.type("text/xml");

  res.send(`
    <Response>
      <Say voice="alice">Esta ligação está sendo gravada.</Say>
      <Record maxLength="3600" />
    </Response>
  `);
});

app.listen(3000, () => {
  console.log("Servidor rodando na porta 3000");
});