const abalone = require("./data/abalone.json")

function sexToNumber(sex) {
  switch (sex) {
    case 'F': return 0
    case 'M': return 1
    default: return 0.5
  }
}

function prepareData(data, ratio = 29) {
  return data.map(row => {
    const values = Object.values(row).slice(0, -1);
    values[0] = sexToNumber(values[0])
    return { input: values, output: [row.rings / ratio] }
  })
}

const shuffle = (arr) => arr.sort(() => Math.random() - .5)
const split = (arr, trainRatio = .75) => {
  const l = Math.floor(arr.length * trainRatio)
  return { train: arr.slice(0, l), test: arr.slice(l) }
}

const prepared = split(shuffle(prepareData(abalone)))
console.log(prepared.train.length)
console.log(prepared.test.length);
