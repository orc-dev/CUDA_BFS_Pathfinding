/**
 * @file          mapVisulaizer.js
 * @brief         Reads a map file and path files, then visualizes the data in 
 *                the browser.
 *
 * @author        Xin Cai
 * @email         xcai72@wisc.edu
 * @date          Nov. 26, 2023
 *
 * @course        ME759: High Performance Computing for Engineering Application
 * @instructor    Professor Dan Negrut
 * @assignment    Final Project
 *
 */
export {};

// global variables and meta data
const WD_SVG = 5000;
const HT_SVG = 9000;

const mapFile   = '../maps/map_512_dense.csv';
const path4File = '../out/gpu_out/path_D4_DW_plus_0.csv';
const path6File = '../out/gpu_out/path_D6_DW_plus_0.csv';

// define the root svg element
const svg = d3.select('body')
    .append('svg')
    .attr('width', WD_SVG)
    .attr('height', HT_SVG)
    .attr('id', 'mysvg');

// read map and path files, then rendering
//  - credits: importing-data-from-multiple-csv-files-in-d3
//  - URL:https://stackoverflow.com/questions/21842384
d3.queue()
    .defer(d3.csv, mapFile)
    .defer(d3.csv, path4File)
    .defer(d3.csv, path6File)
    .await(function(error, mapData, path4Data, path6Data) {
        if (error)
            console.error(`Caught error:\n${error}`);
        else
            renderMap(mapData, path4Data, path6Data);
    });

/**
 * Render the map with square-cell and hexagon-cell,
 * highlighting paths on both 4-way and 6-way maps.
 * 
 * @param {Object} mapData data from map file
 * @param {Object} path4Data shortest path data for 4-way map
 * @param {Object} path6Data shortest path data for 6-way map
 */
function renderMap(mapData, path4Data, path6Data) {
    // preprocess data
    const MAP_NROW = mapData.length;
    if (MAP_NROW == 0)
        console.log('Warning: map data is empty.');

    if (Object.keys(path4Data).length > 0)
        delete path4Data['columns'];

    if (Object.keys(path6Data).length > 0)
        delete path6Data['columns'];

    // color data
    const cellColor = [toHEXC([220, 220, 220]),   // open cell
                       toHEXC([116, 128, 126])];  // obstacle
    const pathColor = toHEXC([200, 50, 50]);      // path
    const pathOpacity = 0.7;

    // geometric data
    const ref_4     = [100, 100];
    const ref_6     = [100, 5000];
    const cellGap   = 1;
    const sqrSize   = 8;
    const hexMaxR   = 4.6;

    const SIN_60    = Math.sin(Math.PI / 3);
    const hexMinR   = SIN_60 * hexMaxR;
    const hexPoints = generateHexagonPoints(hexMaxR);
    const horiStep  = hexMinR * 2 + cellGap;
    const vertStep  = horiStep * SIN_60;
    const offsetHoriOddRow = horiStep / 2;
    
    // build path points from path data
    const path4 = Object.values(path4Data).map(p => [Number(p.x), Number(p.y)]);
    const path6 = Object.values(path6Data).map(p => [Number(p.x), Number(p.y)]);

    for (let p of path4) {
        console.log(`${p[0]}, ${p[1]}`);
    }

    // rendering maps
    renderMapWithSquare(...ref_4);
    renderMapwithHexagon(...ref_6);
    // rendering paths
    renderPath4(...ref_4);
    renderPath6(...ref_6);

    function renderMapWithSquare(ox, oy) {
        const step = sqrSize + cellGap;
        for (let r = 0; r < MAP_NROW;  ++r) {
            for (let c = 0; c < MAP_NROW; ++c) {
                const k = mapData[r][c];
                const x = ox + c * step;
                const y = oy + r * step;
                drawSquare(x, y, cellColor[k], 1);
            }
        }
    }

    function renderMapwithHexagon(ox, oy) {
        for (let r = 0; r < MAP_NROW; ++r) {
            for (let c = 0; c < MAP_NROW; ++c) {
                const dx = (r & 1) * offsetHoriOddRow;
                const x  = (c * horiStep) + ox + dx;
                const y  = (r * vertStep) + oy;
                const sixPoints = hexPoints.map(e => [e[0] + x, e[1] + y]);
                drawHexagon(sixPoints, cellColor[mapData[r][c]], 1);
            }
        }
    }

    function renderPath4(ox, oy) {
        for (let p of path4) {
            const x = p[1] * (sqrSize + cellGap) + ox;
            const y = p[0] * (sqrSize + cellGap) + oy;
            drawSquare(x, y, pathColor, pathOpacity);
        }
    }

    function renderPath6(ox, oy) {
        for (let p of path6) {
            const dx = (p[0] & 1) * offsetHoriOddRow;
            const x  = (p[1] * horiStep) + ox + dx;
            const y  = (p[0] * vertStep) + oy;
            const sixPoints = hexPoints.map(e => [e[0] + x, e[1] + y]);
            drawHexagon(sixPoints, pathColor, pathOpacity);
        }
    }

    function drawSquare(x, y, color, opacity) {
        svg.append('rect')
            .attr('x', x)
            .attr('y', y)
            .attr('width', sqrSize)
            .attr('height', sqrSize)
            .attr('fill', color)
            .attr('fill-opacity', opacity);
    }

    function drawHexagon(sixPoints, color, opacity) {
        svg.append('path')
            .attr('d', 'M' + sixPoints.join('L') + 'Z')
            .attr('fill', color)
            .attr('fill-opacity', opacity);
    }

    function generateHexagonPoints(radius) {
        const points = [];
        for (let i = 0; i < 6; i++) {
            const angle = (i * Math.PI + Math.PI / 2) / 3;
            const x = radius * Math.cos(angle);
            const y = radius * Math.sin(angle);
            points.push([x, y]);
        }
        return points;
    }

    function toHEXC(arr) {
        return arr.map(e => e.toString(16).padStart(2, '0'))
                  .reduce((p, c) => p + c, '#');
    }
}
