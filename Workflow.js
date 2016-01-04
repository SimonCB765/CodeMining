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

// Define dataset size variables.
var rawDatasetDimensions = { height : 150, width : 100 };
var processedDatasetDimensions = { height : rawDatasetDimensions.height * 2 / 3, width : rawDatasetDimensions.width };
var unusedDatasetDimensions = { height : rawDatasetDimensions.height / 3, width : rawDatasetDimensions.width };

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