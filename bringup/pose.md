
## 预抓取位姿

```bash
curl -X POST http://localhost:5000/pose \
  -H "Content-Type: application/json" \
  -d '{"arr":[0.47496347980986053,-0.15853520985100475,0.3,-0.998817958645,0.039610922508,-0.024688541864,0.013569679729]}'
```

## 抓取位姿

```bash
curl -X POST http://localhost:5000/pose \
  -H "Content-Type: application/json" \
  -d '{"arr":[0.5223588564835688,-0.2088280995192223,0.02,-0.999591888726,-0.001538639162,-0.028355211322,0.003109754664]}'
```

## 底

```bash
curl -X POST http://localhost:5000/pose \
  -H "Content-Type: application/json" \
  -d '{"arr":[0.5249042791397374,-0.2033333375850275,0.02,0.999483915054,-0.008555643087,0.030376872331,0.00599584423]}'
```

## 左上

```bash
curl -X POST http://localhost:5000/pose \
  -H "Content-Type: application/json" \
  -d '{"arr":[0.6494557798218754,0.09862840778854759,0.02,0.9998406911220439,-0.0066598737201470646,0.01586298560234512,-0.004754381811027756]}'
```

## 右上

```bash
curl -X POST http://localhost:5000/pose \
  -H "Content-Type: application/json" \
  -d '{"arr":[0.6518036916757318,-0.43203588800659815,0.02,0.9992485813200739,-0.014504048968225594,0.03391831978060313,0.011893396344520938]}'
```

## 右下

```bash
curl -X POST http://localhost:5000/pose \
  -H "Content-Type: application/json" \
  -d '{"arr":[0.4076757559051058,-0.4221984673398182,0.02,0.9995843170176683,-0.028038389035206213,0.0031921480049484025,-0.005903567035837695]}'
```

## 左下

```bash
curl -X POST http://localhost:5000/pose \
  -H "Content-Type: application/json" \
  -d '{"arr":[0.41668718594322274,0.06765306085258509,0.02,0.9991961837520459,-0.02023461128564343,-0.0289737121926385,-0.018922760983659737]}'
```

夹爪命令可以这样配合：

```bash
curl -X POST http://localhost:5000/open_gripper
curl -X POST http://localhost:5000/close_gripper
```
