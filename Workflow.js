svgWidth = 1000;  // The width of the SVG element containing the diagram.
svgHeight = 800;  // The height of the SVG element containing the diagram.

// Create the SVG element.
var svg = d3.select("body")
    .append("svg")
        .attr("width", svgWidth)
        .attr("height", svgHeight);

// Define the arrowhead shape.
var arrowheadWidth = 6;
var arrowheadHeight = 4;
var definitions = svg.append("defs");
var arrowhead = definitions.append("marker")
    .attr("id", "arrowhead")
    .attr("refX", 0)  // Set refX so that the left edge of the marker is affixed to the end of the arrow's line.
    .attr("refY", arrowheadHeight / 2)  // Set refY so that the middle of the marker (vertically) is affixed to the end of the arrow's line.
    .attr("markerWidth", arrowheadWidth)
    .attr("markerHeight", arrowheadHeight)
    .attr("orient", "auto");
arrowhead.append("path")
    .attr("d", "M0,0V" + arrowheadHeight + "L" + arrowheadWidth + "," + (arrowheadHeight / 2) + "Z");

// Define positioning variables.
var hGapBetweenPictures = 100;
var yOffset = 200;
var xOffset = 10;

// Define dataset and model size variables.
var rawDatasetDimensions = { height : 150, width : 100 };
var processedDatasetDimensions = { height : rawDatasetDimensions.height * 2 / 3, width : rawDatasetDimensions.width };
var unusedDatasetDimensions = { height : rawDatasetDimensions.height / 3, width : rawDatasetDimensions.width };
var initialModelDimensions = processedDatasetDimensions

// Create the raw data picture.
var rawDataPosition = { x : xOffset, y : yOffset };
var containerRawData = svg.append("g")
    .classed("rawData", true)
    .attr("transform", "translate(" + rawDataPosition.x + ", " + rawDataPosition.y + ")");
createTable(containerRawData, 0, 0, rawDatasetDimensions.width, rawDatasetDimensions.height, "Raw Data");

// Create the processed data picture.
xOffset += rawDatasetDimensions.width + hGapBetweenPictures;
var processedDataPosition = { x : xOffset, y : yOffset - (rawDatasetDimensions.height / 10) };
var containerProcessedData = svg.append("g")
    .classed("processedData", true)
    .attr("transform", "translate(" + processedDataPosition.x + ", " + processedDataPosition.y + ")");
createTable(containerProcessedData, 0, 0, processedDatasetDimensions.width, processedDatasetDimensions.height, "Processed Data");

// Create the unused data picture.
var unusedDataPosition = { x : xOffset, y : yOffset + (rawDatasetDimensions.height * 7 / 6) - unusedDatasetDimensions.height };
var containerProcessedData = svg.append("g")
    .classed("unusedData", true)
    .attr("transform", "translate(" + unusedDataPosition.x + ", " + unusedDataPosition.y + ")");
createTable(containerProcessedData, 0, 0, unusedDatasetDimensions.width, unusedDatasetDimensions.height, "Unused Data");

// Create the arrows from the raw to processed and unused data.
var arrowStartX = rawDataPosition.x + rawDatasetDimensions.width;
var arrowStartY = rawDataPosition.y + (rawDatasetDimensions.height / 2);
createArrow(svg, arrowStartX, arrowStartY, processedDataPosition.x, processedDataPosition.y + (processedDatasetDimensions.height / 2));
createArrow(svg, arrowStartX, arrowStartY, unusedDataPosition.x, unusedDataPosition.y + (unusedDatasetDimensions.height / 2));

// Create the initial model.
xOffset += processedDatasetDimensions.width + hGapBetweenPictures;
var initialModelPosition = { x : xOffset, y : processedDataPosition.y };
var containerInitialModel = svg.append("g")
    .classed("initialModel", true)
    .attr("transform", "translate(" + initialModelPosition.x + ", " + initialModelPosition.y + ")");
createModel(containerInitialModel, 0, 0, initialModelDimensions.width, initialModelDimensions.height, "Initial Model");

function createArrow(selection, startX, startY, endX, endY)
{
    var arrow = selection.append("path")
        .classed("arrow", true);
    endX -= (arrowheadWidth * parseInt(arrow.style("stroke-width")));  // Determine the size of the arrowhead and move the end of the arrow's line as needed.
    var contrl1 = { x : startX + ((endX - startX) / 2), y : startY };
    var contrl2 = { x : startX + ((endX - startX) / 2), y : endY };
    arrow
        .attr("d", "M" + startX + "," + startY + "C" + contrl1.x + "," + contrl1.y + "," + contrl2.x + "," + contrl2.y + "," + endX + "," + endY)
        .attr("marker-end", "url(#arrowhead)");
}

function createModel(selection, x, y, width, height, text)
{
    selection.append("rect")
        .attr("class", "modelEdge")
        .attr("x", x)
        .attr("y", y)
        .attr("width", width)
        .attr("height", height);

    selection.append("rect")
        .attr("class", "modelInternal")
        .attr("x", x)
        .attr("y", y)
        .attr("width", width)
        .attr("height", height);

    // Add text.
    var textToAdd = selection.append("text")
        .attr("text-anchor", "middle")
        .attr("x", width / 2)
        .text(text);
    var textBBox = textToAdd.node().getBBox();
    textToAdd.attr("y", y - (textBBox.height * 1 / 2));

    // Add data points.
    var data = [ { x : 10, y : 10 }, { x : 20, y : 20 }, { x : 30, y : 30 }, { x : 40, y : 40 }, { x : 50, y : 50 }, { x : 60, y : 60 }, { x : 70, y : 70 } ];
    var datapoints = selection.selectAll("circle")
        .data(data)
            .enter()
            .append("circle")
                .classed("modelData", true)
                .attr("cx", function(d) { return d.x; })
                .attr("cy", function(d) { return d.y; })
                .attr("r", 3);

    // Add the line.
    var start = { x : x + (width / 10), y : y + (height * 9 / 10) };
    var end = { x : x + (width * 9 / 10), y : y + (height / 10) };
    var control1 = { x : 40, y : 40 };
    var control2 = { x : 50, y : 50 };
    var fitLine = selection.append("path")
        .classed("modeLine", true)
        .attr("d", "M" + start.x + "," + start.y + "C" + control1.x + "," + control2.y + "," + control2.x + "," + control2.y + "," + end.x + "," + end.y);
}

function createTable(selection, x, y, width, height, text)
{
    selection.append("rect")
        .attr("class", "tableEdge")
        .attr("x", x)
        .attr("y", y)
        .attr("width", width)
        .attr("height", height);

    selection.append("rect")
        .attr("class", "tableInternal")
        .attr("x", x)
        .attr("y", y)
        .attr("width", width)
        .attr("height", height);

    // Add text.
    var textToAdd = selection.append("text")
        .attr("text-anchor", "middle")
        .attr("x", width / 2)
        .text(text);
    var textBBox = textToAdd.node().getBBox();
    textToAdd.attr("y", y - (textBBox.height * 1 / 2));

    // Add rows.
    var numberRows = 1;
    while ((height / numberRows) > 10)
    {
        numberRows++;
    }
    var rowHeight = height / numberRows;
    for (var i = 1; i < numberRows; i++)
    {
        selection.append("line")
            .attr("class", "tableLine")
            .attr("x1", x)
            .attr("y1", y + (i * rowHeight))
            .attr("x2", x + width)
            .attr("y2", y + (i * rowHeight));
    }

    // Add columns.
    var numberCols = 1;
    while ((width / numberCols) > 20)
    {
        numberCols++;
    }
    var colWidth = width / numberCols;
    for (var i = 1; i < numberCols; i++)
    {
        selection.append("line")
            .attr("class", "tableLine")
            .attr("x1", i * colWidth)
            .attr("y1", y)
            .attr("x2", i * colWidth)
            .attr("y2", y + height);
    }
}