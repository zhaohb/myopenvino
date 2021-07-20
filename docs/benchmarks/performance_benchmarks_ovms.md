# OpenVINO™ Model Server Benchmark Results {#openvino_docs_performance_benchmarks_ovms}

OpenVINO™ Model Server is an open-source, production-grade inference platform that exposes a set of models via a convenient inference API over gRPC or HTTP/REST. It employs the inference engine libraries for from the Intel® Distribution of OpenVINO™ toolkit to extend workloads across Intel® hardware including CPU, GPU and others.

![OpenVINO™ Model Server](../img/performance_benchmarks_ovms_01.png)

## Measurement Methodology

OpenVINO™ Model Server is measured in multiple-client-single-server configuration using two hardware platforms connected by ethernet network. The network bandwidth depends on the platforms as well as models under investigation and it is set to not be a bottleneck for workload intensity. This connection is dedicated only to the performance measurements. The benchmark setup is consists of four main parts:

![OVMS Benchmark Setup Diagram](../img/performance_benchmarks_ovms_02.png)

* **OpenVINO™ Model Server** is launched as a docker container on the server platform and it listens (and answers on) requests from clients. OpenVINO™ Model Server is run on the same machine as the OpenVINO™ toolkit benchmark application in corresponding benchmarking. Models served by OpenVINO™ Model Server are located in a local file system mounted into the docker container. The OpenVINO™ Model Server instance communicates with other components via ports over a dedicated docker network.

* **Clients** are run in separated physical machine referred to as client platform. Clients are implemented in Python3 programming language based on TensorFlow* API and they work as parallel processes. Each client waits for a response from OpenVINO™ Model Server before it will send a new next request. The role played by the clients is also verification of responses.

* **Load balancer** works on the client platform in a docker container. HAProxy is used for this purpose. Its main role is counting of requests forwarded from clients to OpenVINO™ Model Server, estimating its latency, and sharing this information by Prometheus service. The reason of locating the load balancer on the client site is to simulate real life scenario that includes impact of physical network on reported metrics.

* **Execution Controller** is launched on the client platform. It is responsible for synchronization of the whole measurement process, downloading metrics from the load balancer, and presenting the final report of the execution.

## 3D U-Net (FP32)
![](../img/throughput_ovms_3dunet.png)
## resnet-50-TF (INT8)
![](../img/throughput_ovms_resnet50_int8.png)
## resnet-50-TF (FP32)
![](../img/throughput_ovms_resnet50_fp32.png)
## bert-large-uncased-whole-word-masking-squad-int8-0001 (INT8)
![](../img/throughput_ovms_bertlarge_int8.png)

## bert-large-uncased-whole-word-masking-squad-0001 (FP32)
![](../img/throughput_ovms_bertlarge_fp32.png)
## Platform Configurations

OpenVINO™ Model Server performance benchmark numbers are based on release 2021.3. Performance results are based on testing as of March 15, 2021 and may not reflect all publicly available updates. 

**Platform with Intel® Xeon® Gold 6252**

<table>
  <tr>
    <th></th>
    <th><strong>Server Platform</strong></th>
    <th><strong>Client Platform</strong></th>
  </tr>
  <tr>
    <td><strong>Motherboard</strong></td>
    <td>Intel® Server Board S2600WF H48104-872</td>
    <td>Inspur YZMB-00882-104 NF5280M5</td>
  </tr>
  <tr>
    <td><strong>Memory</strong></td>
    <td>Hynix 16 x 16GB @ 2666 MT/s DDR4</td>
    <td>Samsung 16 x 16GB @ 2666 MT/s DDR4</td>
  </tr>
  <tr>
    <td><strong>CPU</strong></td>
    <td>Intel® Xeon® Gold 6252 CPU @ 2.10GHz</td>
    <td>Intel® Xeon® Platinum 8260M CPU @ 2.40GHz</td>
  </tr>
  <tr>
    <td><strong>Selected CPU Flags</strong></td>
    <td>Hyper Threading, Turbo Boost, DL Boost</td>
    <td>Hyper Threading, Turbo Boost, DL Boost</td>
  </tr>
  <tr>
    <td><strong>CPU Thermal Design Power</strong></td>
    <td>150 W</td>
    <td>162 W</td>
  </tr>
  <tr>
    <td><strong>Operating System</strong></td>
    <td>Ubuntu 20.04.2 LTS</td>
    <td>Ubuntu 20.04.2 LTS</td>
  </tr>
  <tr>
    <td><strong>Kernel Version</strong></td>
    <td>5.4.0-65-generic</td>
    <td>5.4.0-54-generic</td>
  </tr>
  <tr>
    <td><strong>BIOS Vendor</strong></td>
    <td>Intel® Corporation</td>
    <td>American Megatrends Inc.</td>
  </tr>
  <tr>
    <td><strong>BIOS Version and Release Date</strong></td>
    <td>SE5C620.86B.02.01, date: 03/26/2020</td>
    <td>4.1.16, date: 06/23/2020</td>
  </tr>
  <tr>
    <td><strong>Docker Version</strong></td>
    <td>20.10.3</td>
    <td>20.10.3</td>
  </tr>
  <tr>
    <td><strong>Network Speed</strong></td>
    <td colspan="2" align="center">40 Gb/s</td>
  </tr>
</table>

**Platform with Intel® Core™ i9-10920X**

<table>
<tr>
  <th></th>
  <th><strong>Server Platform</strong></th>
  <th><strong>Client Platform</strong></th>
</tr>
<tr>
  <td><strong>Motherboard</strong></td>
  <td>ASUSTeK COMPUTER INC. PRIME X299-A II</td>
  <td>ASUSTeK COMPUTER INC. PRIME Z370-P</td>
</tr>
<tr>
  <td><strong>Memory</strong></td>
  <td>Corsair 4 x 16GB @ 2666 MT/s DDR4</td>
  <td>Corsair 4 x 16GB @ 2133 MT/s DDR4</td>
</tr>
<tr>
  <td><strong>CPU</strong></td>
  <td>Intel® Core™ i9-10920X CPU @ 3.50GHz</td>
  <td>Intel® Core™ i7-8700T CPU @ 2.40GHz</td>
</tr>
<tr>
  <td><strong>Selected CPU Flags</strong></td>
  <td>Hyper Threading, Turbo Boost, DL Boost</td>
  <td>Hyper Threading, Turbo Boost</td>
</tr>
<tr>
  <td><strong>CPU Thermal Design Power</strong></td>
  <td>165 W</td>
  <td>35 W</td>
</tr>
<tr>
  <td><strong>Operating System</strong></td>
  <td>Ubuntu 20.04.1 LTS</td>
  <td>Ubuntu 20.04.1 LTS</td>
</tr>

<tr>
  <td><strong>Kernel Version</strong></td>
  <td>5.4.0-52-generic</td>
  <td>5.4.0-56-generic</td>
</tr>
<tr>
  <td><strong>BIOS Vendor</strong></td>
  <td>American Megatrends Inc.</td>
  <td>American Megatrends Inc.</td>
</tr>
<tr>
  <td><strong>BIOS Version and Release Date</strong></td>
  <td>0603, date: 03/05/2020</td>
  <td>2401, date: 07/15/2019</td>
</tr>
<tr>
  <td><strong>Docker Version</strong></td>
  <td>19.03.13</td>
  <td>19.03.14</td>
</tr>
</tr>
<tr>
  <td><strong>Network Speed</strong></td>
  <td colspan="2" align="center">10 Gb/s</td>
</tr>
</table>

**Platform with Intel® Core™ i7-8700T**

<table>
<tr>
  <th></th>
  <th><strong>Server Platform</strong></th>
  <th><strong>Client Platform</strong></th>
</tr>
<tr>
  <td><strong>Motherboard</strong></td>
  <td>ASUSTeK COMPUTER INC. PRIME Z370-P</td>
  <td>ASUSTeK COMPUTER INC. PRIME X299-A II</td>
</tr>
<tr>
  <td><strong>Memory</strong></td>
  <td>Corsair 4 x 16GB @ 2133 MT/s DDR4</td>
  <td>Corsair 4 x 16GB @ 2666 MT/s DDR4</td>
</tr>
<tr>
  <td><strong>CPU</strong></td>
  <td>Intel® Core™ i7-8700T CPU @ 2.40GHz</td>
  <td>Intel® Core™ i9-10920X CPU @ 3.50GHz</td>
</tr>
<tr>
  <td><strong>Selected CPU Flags</strong></td>
  <td>Hyper Threading, Turbo Boost</td>
  <td>Hyper Threading, Turbo Boost, DL Boost</td>
</tr>
<tr>
  <td><strong>CPU Thermal Design Power</strong></td>
  <td>35 W</td>
  <td>165 W</td>
</tr>
<tr>
  <td><strong>Operating System</strong></td>
  <td>Ubuntu 20.04.1 LTS</td>
  <td>Ubuntu 20.04.1 LTS</td>
</tr>

<tr>
  <td><strong>Kernel Version</strong></td>
  <td>5.4.0-56-generic</td>
  <td>5.4.0-52-generic</td>
</tr>
<tr>
  <td><strong>BIOS Vendor</strong></td>
  <td>American Megatrends Inc.</td>
  <td>American Megatrends Inc.</td>
</tr>
<tr>
  <td><strong>BIOS Version and Release Date</strong></td>
  <td>2401, date: 07/15/2019</td>
  <td>0603, date: 03/05/2020</td>
</tr>
<tr>
  <td><strong>Docker Version</strong></td>
  <td>19.03.14</td>
  <td>19.03.13</td>
</tr>
</tr>
<tr>
  <td><strong>Network Speed</strong></td>
  <td colspan="2" align="center">10 Gb/s</td>
</tr>
</table>

**Platform with Intel® Core™ i5-8500**

<table>
<tr>
  <th></th>
  <th><strong>Server Platform</strong></th>
  <th><strong>Client Platform</strong></th>
</tr>
<tr>
  <td><strong>Motherboard</strong></td>
  <td>ASUSTeK COMPUTER INC. PRIME Z370-A</td>
  <td>Gigabyte Technology Co., Ltd. Z390 UD</td>
</tr>
<tr>
  <td><strong>Memory</strong></td>
  <td>Corsair 2 x 16GB @ 2133 MT/s DDR4</td>
  <td>029E 4 x 8GB @ 2400 MT/s DDR4</td>
</tr>
<tr>
  <td><strong>CPU</strong></td>
  <td>Intel® Core™ i5-8500 CPU @ 3.00GHz</td>
  <td>Intel® Core™ i3-8100 CPU @ 3.60GHz</td>
</tr>
<tr>
  <td><strong>Selected CPU Flags</strong></td>
  <td>Turbo Boost</td>
  <td>-</td>
</tr>
<tr>
  <td><strong>CPU Thermal Design Power</strong></td>
  <td>65 W</td>
  <td>65 W</td>
</tr>
<tr>
  <td><strong>Operating System</strong></td>
  <td>Ubuntu 20.04.1 LTS</td>
  <td>Ubuntu 20.04.1 LTS</td>
</tr>
<tr>
  <td><strong>Kernel Version</strong></td>
  <td>5.4.0-52-generic</td>
  <td>5.4.0-52-generic</td>
</tr>
<tr>
  <td><strong>BIOS Vendor</strong></td>
  <td>American Megatrends Inc.</td>
  <td>American Megatrends Inc.</td>
</tr>
<tr>
  <td><strong>BIOS Version and Release Date</strong></td>
  <td>2401, date: 07/12/2019</td>
  <td>F10j, date: 09/16/2020</td>
</tr>
<tr>
  <td><strong>Docker Version</strong></td>
  <td>19.03.13</td>
  <td>20.10.0</td>
</tr>
</tr>
<tr>
  <td><strong>Network Speed</strong></td>
  <td colspan="2" align="center">40 Gb/s</td>
</tr>
</table>

**Platform with Intel® Core™ i3-8100**
<table>
<tr>
  <th></th>
  <th><strong>Server Platform</strong></th>
  <th><strong>Client Platform</strong></th>
</tr>
<tr>
  <td><strong>Motherboard</strong></td>
  <td>Gigabyte Technology Co., Ltd. Z390 UD</td>
  <td>ASUSTeK COMPUTER INC. PRIME Z370-A</td>
</tr>
<tr>
  <td><strong>Memory</strong></td>
  <td>029E 4 x 8GB @ 2400 MT/s DDR4</td>
  <td>Corsair 2 x 16GB @ 2133 MT/s DDR4</td>
</tr>
<tr>
  <td><strong>CPU</strong></td>
  <td>Intel® Core™ i3-8100 CPU @ 3.60GHz</td>
  <td>Intel® Core™ i5-8500 CPU @ 3.00GHz</td>
</tr>
<tr>
  <td><strong>Selected CPU Flags</strong></td>
  <td>-</td>
  <td>Turbo Boost</td>
</tr>
<tr>
  <td><strong>CPU Thermal Design Power</strong></td>
  <td>65 W</td>
  <td>65 W</td>
</tr>
<tr>
  <td><strong>Operating System</strong></td>
  <td>Ubuntu 20.04.1 LTS</td>
  <td>Ubuntu 20.04.1 LTS</td>
</tr>
<tr>
  <td><strong>Kernel Version</strong></td>
  <td>5.4.0-52-generic</td>
  <td>5.4.0-52-generic</td>
</tr>
<tr>
  <td><strong>BIOS Vendor</strong></td>
  <td>American Megatrends Inc.</td>
  <td>American Megatrends Inc.</td>
</tr>
<tr>
  <td><strong>BIOS Version and Release Date</strong></td>
  <td>F10j, date: 09/16/2020</td>
  <td>2401, date: 07/12/2019</td>
</tr>
<tr>
  <td><strong>Docker Version</strong></td>
  <td>20.10.0</td>
  <td>19.03.13</td>
</tr>
</tr>
<tr>
  <td><strong>Network Speed</strong></td>
  <td colspan="2" align="center">40 Gb/s</td>
</tr>
</table>


\htmlonly
<style>
    .footer {
        display: none;
    }
</style>
<div class="opt-notice-wrapper">
<p class="opt-notice">
\endhtmlonly
Results may vary. For workloads and configurations visit: [www.intel.com/PerformanceIndex](https://www.intel.com/PerformanceIndex) and [Legal Information](../Legal_Information.md).
\htmlonly
</p>
</div>
\endhtmlonly

