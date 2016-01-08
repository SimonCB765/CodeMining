var data = [
            ["Patient ID", "Code1", "Code2", "Code3", "Code4", "..."],
            [1, 1, 0, 1, 1, "..."],
            [2, 1, 0, 1, 0, "..."],
            [3, 0, 0, 0, 1, "..."],
            [4, 0, 0, 1, 1, "..."]
           ];





svgWidth = 800;  // The width of the SVG element containing the diagram.
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

// Create the input data table.
var inputDataCoords = { x : 10, y : 10 };
var inputDataTable = svg.append("g")
    .classed("table", true)
    .attr("transform", "translate(" + inputDataCoords.x + "," + inputDataCoords.y + ")");
create_table(inputDataTable, data, 25, 75, 2);

function create_table(selection, data, cellHeight, cellWidth, headerRows)
{
    // Determine the number of rows and columns needed to fit the data.
    var numberRows = data.length - 1 + headerRows;
    var numberCols = data[0].length;

    // Determine the width and height of the table.
    var width = numberCols * cellWidth;
    var height = numberRows * cellHeight;

    // Create the table borders.
    selection.append("rect")
        .attr("class", "tableEdge")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", width)
        .attr("height", height);
    selection.append("rect")
        .attr("class", "tableInternal")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", width)
        .attr("height", height);

    // Create the rows.
    var rowY = headerRows * cellHeight;  // Skip making any row lines within the header section.
    while (rowY < height)
    {
        selection.append("line")
            .attr("class", "tableLine")
            .attr("x1", 0)
            .attr("y1", rowY)
            .attr("x2", width)
            .attr("y2", rowY);
        rowY += cellHeight;
    }
    
    // Create the columns.
    for (var i = 1; i < numberCols; i++)
    {
        selection.append("line")
            .attr("class", "tableLine")
            .attr("x1", i * cellWidth)
            .attr("y1", 0)
            .attr("x2", i * cellWidth)
            .attr("y2", height);
    }

    // Add text.
    var headerTextPos = { x : cellWidth / 2, y : headerRows * cellHeight / 2 };
    var cellTextPos = { x : cellWidth / 2, y : cellHeight / 2 };
    function text_adder(element, index)
    {
        if (index == 0)
        {
            // Adding the header.
            element.forEach(function(d, i)
            {
                selection.append("text")
                    .attr("text-anchor", "middle")
                    .attr("x", headerTextPos.x + (i * cellWidth))
                    .attr("y", headerTextPos.y)
                    .text(d);
            });
        }
        else
        {
            // Adding a data row.
            element.forEach(function(d, i)
            {
                selection.append("text")
                    .attr("text-anchor", "middle")
                    .attr("x", cellTextPos.x + (i * cellWidth))
                    .attr("y", cellTextPos.y + (headerRows * cellHeight) + ((index - 1) * cellHeight))
                    .text(d);
            });
        }
    }
    data.forEach(text_adder);
}