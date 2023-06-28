# Info

## Settings

### Cloud Storage

File Filter Regex
`Vegetable Images/test/.*(jpe?g|png)`

S3 Endpoint
`http://minio:9000`

Treat every bucket object as a source file
`true`

Recursive scan
`true`

### Labeling Interface

Code

```
<View>
  <Image name="image" value="$image"/>
  <Choices name="choice" toName="image">
    <Choice value="Bean"/>
    <Choice value="Bitter_Gourd"/>
    <Choice value="Bottle_Gourd"/>
    <Choice value="Brinjal"/>
    <Choice value="Broccoli"/>
    <Choice value="Cabbage"/>
    <Choice value="Capsicum"/>
    <Choice value="Carrot"/>
    <Choice value="Cauliflower"/>
    <Choice value="Cucumber"/>
    <Choice value="Papaya"/>
    <Choice value="Potato"/>
    <Choice value="Pumpkin"/>
    <Choice value="Radish"/>
    <Choice value="Tomato"/>
  </Choices>
</View>
```

### Machine Learning

Add Model

URL
`http://serving:8080`

Use for interactive preannotations
`true`
