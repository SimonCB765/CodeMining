svgWidth = 1020;  // The width of the SVG element containing the diagram.
svgHeight = 300;  // The height of the SVG element containing the diagram.

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
var yOffset = 100;
var xOffset = 10;

// Define dataset and model size variables.
var rawDatasetDimensions = { height : 150, width : 100 };
var processedDatasetDimensions = { height : rawDatasetDimensions.height * 2 / 3, width : rawDatasetDimensions.width };
var unusedDatasetDimensions = { height : rawDatasetDimensions.height / 3, width : rawDatasetDimensions.width };
var initialModelDimensions = processedDatasetDimensions;
var correctlyClassifiedDimensions = { height : processedDatasetDimensions.height * 8 / 10, width : processedDatasetDimensions.width };
var misclassifiedDimensions = { height : processedDatasetDimensions.height * 2 / 10, width : processedDatasetDimensions.width };
var finalModelDimensions = initialModelDimensions;

// Create the raw data picture.
var rawDataPosition = { x : xOffset, y : yOffset };
var containerRawData = svg.append("g")
    .classed("rawData", true)
    .attr("transform", "translate(" + rawDataPosition.x + ", " + rawDataPosition.y + ")");
create_table(containerRawData, 0, 0, rawDatasetDimensions.width, rawDatasetDimensions.height, "Processed Data");

// Create the processed data picture.
xOffset += rawDatasetDimensions.width + hGapBetweenPictures;
var processedDataPosition = { x : xOffset, y : yOffset - (rawDatasetDimensions.height / 10) };
var containerProcessedData = svg.append("g")
    .classed("processedData", true)
    .attr("transform", "translate(" + processedDataPosition.x + ", " + processedDataPosition.y + ")");
create_table(containerProcessedData, 0, 0, processedDatasetDimensions.width, processedDatasetDimensions.height, "Training Data");

// Create the unused data picture.
var unusedDataPosition = { x : xOffset, y : yOffset + (rawDatasetDimensions.height * 7 / 6) - unusedDatasetDimensions.height };
var containerProcessedData = svg.append("g")
    .classed("unusedData", true)
    .attr("transform", "translate(" + unusedDataPosition.x + ", " + unusedDataPosition.y + ")");
create_table(containerProcessedData, 0, 0, unusedDatasetDimensions.width, unusedDatasetDimensions.height, "Unused Data");

// Create the arrows from the raw to processed and unused data.
var arrowStartX = rawDataPosition.x + rawDatasetDimensions.width;
var arrowStartY = rawDataPosition.y + (rawDatasetDimensions.height / 2);
create_arrow(svg, arrowStartX, arrowStartY, processedDataPosition.x, processedDataPosition.y + (processedDatasetDimensions.height / 2));
create_arrow(svg, arrowStartX, arrowStartY, unusedDataPosition.x, unusedDataPosition.y + (unusedDatasetDimensions.height / 2));

// Create the initial model picture.
xOffset += processedDatasetDimensions.width + hGapBetweenPictures;
var initialModelPosition = { x : xOffset, y : processedDataPosition.y };
var containerInitialModel = svg.append("g")
    .classed("initialModel", true)
    .attr("transform", "translate(" + initialModelPosition.x + ", " + initialModelPosition.y + ")");
create_model(containerInitialModel, 0, 0, initialModelDimensions.width, initialModelDimensions.height, "Initial Model");

// Create the correctly classified data picture.
xOffset += initialModelDimensions.width + (hGapBetweenPictures * 2);
var correctlyClassifiedPosition = { x : xOffset, y : processedDataPosition.y - (correctlyClassifiedDimensions.height / 2) };
var containerCorrectlyClassified = svg.append("g")
    .classed("processedData", true)
    .attr("transform", "translate(" + correctlyClassifiedPosition.x + ", " + correctlyClassifiedPosition.y + ")");
create_table(containerCorrectlyClassified, 0, 0, correctlyClassifiedDimensions.width, correctlyClassifiedDimensions.height, "Posterior Above\nThreshold");

// Create the misclassified data picture.
var misclassifiedPosition = { x : xOffset, y : correctlyClassifiedPosition.y + correctlyClassifiedDimensions.height + (misclassifiedDimensions.height * 2) };
var containerMisclassified = svg.append("g")
    .classed("unusedData", true)
    .attr("transform", "translate(" + misclassifiedPosition.x + ", " + misclassifiedPosition.y + ")");
create_table(containerMisclassified, 0, 0, misclassifiedDimensions.width, misclassifiedDimensions.height, "Discarded");

// Create the arrows going in and out of the initial model.
var processedToInitialArrow = { startX : processedDataPosition.x + processedDatasetDimensions.width,
                                startY : processedDataPosition.y + (processedDatasetDimensions.height / 2),
                                endX : initialModelPosition.x,
                                endY : initialModelPosition.y + (initialModelDimensions.height / 2)
                              };
var initialToCorrectArrow = { startX : initialModelPosition.x + initialModelDimensions.width + hGapBetweenPictures,
                              startY : initialModelPosition.y + (initialModelDimensions.height / 2),
                              endX : correctlyClassifiedPosition.x,
                              endY : correctlyClassifiedPosition.y + (correctlyClassifiedDimensions.height / 2)
                            };
var initialToMisclassifiedArrow = { startX : initialModelPosition.x + initialModelDimensions.width + hGapBetweenPictures,
                                    startY : initialModelPosition.y + (initialModelDimensions.height / 2),
                                    endX : misclassifiedPosition.x,
                                    endY : misclassifiedPosition.y + (misclassifiedDimensions.height / 2)
                                  };
create_arrow(svg, processedToInitialArrow.startX, processedToInitialArrow.startY, processedToInitialArrow.endX, processedToInitialArrow.endY);
create_arrow(svg, initialToCorrectArrow.startX, initialToCorrectArrow.startY, initialToCorrectArrow.endX, initialToCorrectArrow.endY);
svg.append("path")
    .classed("arrow", true)
    .attr("d", "M" + (initialModelPosition.x + initialModelDimensions.width) + "," + (initialModelPosition.y + (initialModelDimensions.height / 2)) + "h" + hGapBetweenPictures);
create_arrow(svg, initialToMisclassifiedArrow.startX, initialToMisclassifiedArrow.startY, initialToMisclassifiedArrow.endX, initialToMisclassifiedArrow.endY);

// Create the line looping under the initial model.
var start = { x : processedToInitialArrow.startX, y : processedToInitialArrow.startY };
var secondArcEnd = { x : initialModelPosition.x, y : initialModelPosition.y + initialModelDimensions.height + 30 };
var firstArcEnd = { x : processedToInitialArrow.startX + (hGapBetweenPictures * 2 / 4), y : start.y + ((secondArcEnd.y - start.y) / 2) };
var controlLeft = { x : firstArcEnd.x, y : start.y };
var thirdArcEnd = { x : secondArcEnd.x + initialModelDimensions.width + (hGapBetweenPictures * 2 / 4), y : firstArcEnd.y };
var controlRight = { x : thirdArcEnd.x, y : secondArcEnd.y };
var fourthArcEnd = { x : secondArcEnd.x + initialModelDimensions.width + hGapBetweenPictures, y : start.y };
svg.append("path")
    .classed("arrow", true)
    .attr("d", "M" + start.x + "," + start.y +  // Starting position.
               "Q" + controlLeft.x + "," + controlLeft.y + "," + firstArcEnd.x + "," + firstArcEnd.y +  // Left top curve.
               "T" + secondArcEnd.x + "," + secondArcEnd.y +  // Left bottom curve.
               "h" + initialModelDimensions.width +
               "Q" + controlRight.x + "," + controlRight.y + "," + thirdArcEnd.x + "," + thirdArcEnd.y +  // Right bottom curve.
               "T" + fourthArcEnd.x + "," + fourthArcEnd.y  // Right top curve.
               );

// Add the final model picture.
var finalModelPosition = { x : correctlyClassifiedPosition.x + correctlyClassifiedDimensions.width + hGapBetweenPictures,
                           y : correctlyClassifiedPosition.y + (correctlyClassifiedDimensions.height / 2) - (finalModelDimensions.height / 2)
                         };
var containerFinalModel = svg.append("g")
    .classed("initialModel", true)
    .attr("transform", "translate(" + finalModelPosition.x + ", " + finalModelPosition.y + ")");
create_model(containerFinalModel, 0, 0, finalModelDimensions.width, finalModelDimensions.height, "Final Model");

// Add the arrow to the final model picture.
var arrowToFinalModel = { startX : correctlyClassifiedPosition.x + correctlyClassifiedDimensions.width,
                          endX : correctlyClassifiedPosition.x + correctlyClassifiedDimensions.width + hGapBetweenPictures,
                          y : correctlyClassifiedPosition.y + (correctlyClassifiedDimensions.height / 2)
                        };
create_arrow(svg, arrowToFinalModel.startX, arrowToFinalModel.y, arrowToFinalModel.endX, arrowToFinalModel.y);

function create_arrow(selection, startX, startY, endX, endY)
{
    // Add an arrow to selection starting at (startX, startY) and ending at (endX, endY). The tip of the arrow will be at (endX, endY), the
    // size of the arrow head is taken into account when positioning the arrow.

    var arrow = selection.append("path")
        .classed("arrow", true);
    endX -= (arrowheadWidth * parseInt(arrow.style("stroke-width")));  // Determine the size of the arrowhead and move the end of the arrow's line as needed.
    var contrl1 = { x : startX + ((endX - startX) / 2), y : startY };
    var contrl2 = { x : startX + ((endX - startX) / 2), y : endY };
    arrow
        .attr("d", "M" + startX + "," + startY + "C" + contrl1.x + "," + contrl1.y + "," + contrl2.x + "," + contrl2.y + "," + endX + "," + endY)
        .attr("marker-end", "url(#arrowhead)");
}

function create_model(selection, x, y, width, height, text)
{
    // Create a representative picture of a model in selection. The upper left corner of the model border will be at (x, y), with the bottom
    // right corner at (x+width, y+height). The title over the model is supplied as the text argument.

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
    var data = [ { x : x + (width * 05 / 100), y : y + (height * 67 / 100) }, { x : x + (width * 10 / 100), y : y + (height * 33 / 100) },
                 { x : x + (width * 15 / 100), y : y + (height * 42 / 100) }, { x : x + (width * 20 / 100), y : y + (height * 86 / 100) },
                 { x : x + (width * 25 / 100), y : y + (height * 76 / 100) }, { x : x + (width * 30 / 100), y : y + (height * 13 / 100) },
                 { x : x + (width * 35 / 100), y : y + (height * 61 / 100) }, { x : x + (width * 40 / 100), y : y + (height * 30 / 100) },
                 { x : x + (width * 45 / 100), y : y + (height * 52 / 100) }, { x : x + (width * 50 / 100), y : y + (height * 71 / 100) },
                 { x : x + (width * 55 / 100), y : y + (height * 07 / 100) }, { x : x + (width * 60 / 100), y : y + (height * 33 / 100) },
                 { x : x + (width * 65 / 100), y : y + (height * 82 / 100) }, { x : x + (width * 70 / 100), y : y + (height * 46 / 100) },
                 { x : x + (width * 75 / 100), y : y + (height * 76 / 100) }, { x : x + (width * 80 / 100), y : y + (height * 13 / 100) },
                 { x : x + (width * 85 / 100), y : y + (height * 61 / 100) }, { x : x + (width * 90 / 100), y : y + (height * 30 / 100) },
                 { x : x + (width * 95 / 100), y : y + (height * 52 / 100) }
               ];
    var datapoints = selection.selectAll("circle")
        .data(data)
            .enter()
            .append("circle")
                .classed("modelData", true)
                .attr("cx", function(d) { return d.x; })
                .attr("cy", function(d) { return d.y; })
                .attr("r", 3);

    // Add the line.
    var start = { x : x + (width * 2 / 100), y : y + (height / 15) };
    var end = { x : x + (width * 99 / 100), y : y + (height * 14 / 15) };
    var control1 = { x : x + (width * 1 / 5), y : y + (height / 2) };
    var control2 = { x : x + (width * 4 / 5), y : y + (height / 2) };
    var fitLine = selection.append("path")
        .classed("modeLine", true)
        .attr("d", "M" + start.x + "," + start.y + "C" + control1.x + "," + control1.y + "," + control2.x + "," + control2.y + "," + end.x + "," + end.y);
}

function create_table(selection, x, y, width, height, text)
{
    // Create a table in selection. The upper left corner of the table will be at (x, y), with the bottom
    // right corner at (x+width, y+height). The title over the table is supplied as the text argument.

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
        .attr("x", width / 2);
    if (text.indexOf("\n") > -1)
    {
        // If the text has a line break in it.
        var chunks = text.split("\n");
        for (i = 0; i < chunks.length; i++)
        {
            textToAdd.append("tspan")
                .attr("x", width / 2)
                .attr("dy", "1em")
                .text(chunks[i]);
        }
        var textBBox = textToAdd.node().getBBox();
        console.log(text);
        console.log(textBBox);
        textToAdd.attr("y", y - (textBBox.height * 1.2));
    }
    else
    {
        textToAdd.text(text);
        var textBBox = textToAdd.node().getBBox();
        console.log(text);
        console.log(textBBox);
        textToAdd.attr("y", y - (textBBox.height * 1 / 2));
    }

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