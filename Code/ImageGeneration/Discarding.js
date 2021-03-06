var figureWidth = 400;  // Height of the entire figure including labels and title.
var figureHeight = 400;  // Width of the entire figure including labels and title.
var figureMargin = {top: 10, right: 10, bottom: 10, left: 10};  // Margin around each individual figure.
var svgWidth = (figureWidth + figureMargin.left + figureMargin.right) * 2;  // Width of the SVG element needed to hold both figures and their padding.
var svgHeight = figureHeight + figureMargin.top + figureMargin.bottom;  // Height of the SVG element needed to hold both figures and their padding.
var svgMargin = {top: 10, right: 10, bottom: 10, left: 10};  // The margin around the set of figures.

// Create the SVG element.
var svg = d3.select("body")
    .append("svg")
        .attr("width", svgWidth + svgMargin.left + svgMargin.right)
        .attr("height", svgHeight + svgMargin.top + svgMargin.bottom)
    .append("g")
        .attr("transform", "translate(" + svgMargin.left + ", " + svgMargin.top + ")");

// Read the data in.
d3.text(dataset, function(text)
    {
        var data = [];
        var rows = text.split("\n");  // Rows of the file.
        var header = rows[0].split("\t");  // The file header.
        var numberClasses = (header.length - 4) / 2;  // Two columns per class, and four unrelated to class.
        var classesUsed = new Set();
        for (i = 1; i < rows.length - 1; i++)
        {
            var chunks = rows[i].split("\t");
            var datum = {};
            datum["PatientID"] = parseInt(chunks[0]);
            datum["Class"] = chunks[1];
            classesUsed.add(chunks[1]);
            for (j = 4; j < 4 + numberClasses; j++)
            {
                var classLabel = header[j].match(/_.*/)[0].substring(1);
                datum[classLabel] = parseFloat(chunks[j]);
            }
            data.push(datum);
        }
        classesUsed = Array.from(classesUsed);

        // Create the left discard graph.
        var discardContainer = svg.append("g").attr("transform", "translate(" + figureMargin.left + ", " + figureMargin.top + ")");
        create_discard_graph(discardContainer, data, classesUsed[0], "(a)", cutoff);

        // Create the right discard graph.
        var xPosRightGraph = figureMargin.left + figureWidth + figureMargin.right + figureMargin.left;
        discardContainer = svg.append("g").attr("transform", "translate(" + xPosRightGraph + ", " + figureMargin.top + ")");
        create_discard_graph(discardContainer, data, classesUsed[1], "(b)", cutoff);
    }
);

function create_discard_graph(figureContainer, dataArray, classToPlot, figureTitle, cutoff)
{
    // Filter out all examples not in the class of interest.
    dataArray = dataArray.filter(function(d) { return (d["Class"] === classToPlot) ? true : false; });

    // Sort in ascending order of posterior of the class of interest.
    dataArray = dataArray.sort(function(a, b) { return d3.ascending(a[classToPlot], b[classToPlot]); });

    // Define positioning variables.
    var figurePadding = {top: 40, right: 5, bottom: 50, left: 60};  // Padding around the graph in the figure to allow for titles and labelling axes.

    // Create scales for the figure.
    var xScale = d3.scale.linear()
        .domain([0, dataArray.length + 1])  // Add 1 to allow for the final datapoint to have space for a line.
        .range([figurePadding.left, figureWidth - figurePadding.right]);
    if (dataArray.length > 5000)
    {
        // Add an axis break on the X axis.
        var breakStart = 2500;
        var breakEnd = 40500;
        var xAxisTicks = break_axis(
            xScale, [0, dataArray.length + 1], [figurePadding.left, figureWidth - figurePadding.right], breakStart,
            breakEnd, rangeDivider=50);
    }
    var yScale = d3.scale.linear()
        .domain([0.0, 1.0])
        .range([figureHeight - figurePadding.bottom, figurePadding.top]);

    // Add the title to the figure.
    var title = figureContainer.append("text")
        .attr("class", "title")
        .attr("text-anchor", "middle")
        .attr("x", ((figureWidth - figurePadding.left - figurePadding.right) / 2) + figurePadding.left)
        .attr("y", figurePadding.top * 3 / 4)
        .text(figureTitle);

    // Add the labels on the axes. To rotate the label on the y axis we use a transformation. As this will alter the coordinate system being used
    // to place the text, we perform the rotation around the point where the text is centered. This causes the text to rotate but not move.
    var xAxisLabel = figureContainer.append("text")
        .attr("class", "label")
        .attr("text-anchor", "middle")
        .attr("x", ((figureWidth - figurePadding.left - figurePadding.right) / 2) + figurePadding.left)
        .attr("y", figureHeight - (figurePadding.bottom / 4))
        .text("Example Index");
    yAxisLabelLoc = { x : figurePadding.left / 3, y : ((figureHeight - figurePadding.top - figurePadding.bottom) / 2) + figurePadding.top };
    var yAxisLabel = figureContainer.append("text")
        .attr("class", "label")
        .attr("text-anchor", "middle")
        .attr("x", yAxisLabelLoc.x)
        .attr("y", yAxisLabelLoc.y)
        .attr("transform", "rotate(-90, " + yAxisLabelLoc.x + ", " + yAxisLabelLoc.y + ")")
        .text("Posterior Probability");

    // Find the index of the first example to have a posterior above 0.05.
    var keptIndex = 0;
    for (i = 0; i < dataArray.length + 1; i++)
    {
        if (dataArray[i][classToPlot] > 0.05)
        {
            keptIndex = i;
            break;
        }
    }

    // Add the discard area. Add this first so that it will not be on top of any of the other lines (discard and axes).
    var discardArea = figureContainer.append("rect")
        .classed("discardArea", true)
        .attr("x", xScale(0))
        .attr("y", yScale(cutoff))
        .attr("width", xScale(keptIndex) - xScale(0))
        .attr("height", yScale(0) - yScale(cutoff));

    // Add the vertical example discard line.
    var discardStart = {"x": xScale(keptIndex), "y": yScale(-0.01)};
    var discardEnd = {"x": xScale(keptIndex), "y": yScale(1.01)};
    var discardPath = "M" + discardStart["x"] + "," + discardStart["y"] + "L" + discardEnd["x"] + "," + discardEnd["y"];
    var discardLine = figureContainer.append("line")
        .classed("discard", true)
        .attr("x1", discardStart["x"])
        .attr("y1", discardStart["y"])
        .attr("x2", discardEnd["x"])
        .attr("y2", discardEnd["y"]);

    // Add the threshold discard line.
    var discardStart = {"x": figurePadding.left - 6, "y": yScale(cutoff)};
    var discardEnd = {"x": figureWidth - figurePadding.right + 6, "y": yScale(cutoff)};
    var discardPath = "M" + discardStart["x"] + "," + discardStart["y"] + "L" + discardEnd["x"] + "," + discardEnd["y"];
    var discardLine = figureContainer.append("line")
        .classed("discard", true)
        .attr("x1", discardStart["x"])
        .attr("y1", discardStart["y"])
        .attr("x2", discardEnd["x"])
        .attr("y2", discardEnd["y"]);
    var cutoffText = figureContainer.append("text")
        .attr("class", "cutoffText")
        .attr("text-anchor", "end")
        .attr("x", discardStart["x"])
        .attr("y", discardStart["y"])
        .attr("dx", -3)
        .attr("dy", ".32em")
        .text(cutoff);

    // Add the line for the class of interest.
    var dataPath = "M" + xScale(0) + "," + yScale(0);
    dataArray.forEach(function(d, index)
    {
        dataPath += "L" + xScale(index) + "," + yScale(d[classToPlot]) + "h1";
    });
    var dataLine = figureContainer.append("path")
        .classed(classToPlot + " line", true)
        .attr("d", dataPath);

    // Add the axes for the figure.
    if (typeof xAxisTicks === 'undefined')
    {
        // If it is undefined then a break has not been added to the axis, as breaking the axis alters the ticks.
        var xAxisTicks = xScale.ticks();
    }
    xAxisTicks.push(keptIndex);  // Add the index of the first kept example to the set of axis ticks.
    var xAxis = d3.svg.axis()
        .scale(xScale)
        .orient("bottom")
        .tickValues(xAxisTicks);
    xAxis = figureContainer.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0, " + (figureHeight - figurePadding.bottom) + ")")
        .call(xAxis)
        .select("text")
        .attr("x", -8);
    var yAxisTicks = yScale.ticks();
    yAxisTicks.splice(yAxisTicks.indexOf(0), 1);  // Remove the 0 from the y axis.
    var yAxis = d3.svg.axis()
        .scale(yScale)
        .orient("left")
        .tickValues(yAxisTicks);
    yAxis = figureContainer.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(" + figurePadding.left + ", 0)")
        .call(yAxis);

    // Add the break indicator to the X axis and the data line if needed.
    if (dataArray.length > 5000)
    {
        var breakLineSeparation = xScale(breakEnd) - xScale(breakStart);
        var breakLineLength = 25;
        var breakLineY = breakLineLength / 5;
        var maskIcon = "M0,0" +
                       "C" + (breakLineLength / 3) + "," + (-breakLineY / 2) + "," + (breakLineLength * 2 / 3) + "," + (breakLineY * 3 / 2) + "," + breakLineLength + "," + breakLineY +
                       "L" + breakLineLength + "," + (breakLineY + breakLineSeparation) +
                       "C" + (breakLineLength * 2 / 3) + "," + ((breakLineY * 3 / 2) + breakLineSeparation) + "," + (breakLineLength / 3) + "," + ((-breakLineY / 2) + breakLineSeparation) + "," + 0 + "," + breakLineSeparation +
                       "Z";
        var breakIcon = "M0,0" +
                        "C" + (breakLineLength / 3) + "," + (-breakLineY / 2) + "," + (breakLineLength * 2 / 3) + "," + (breakLineY * 3 / 2) + "," + breakLineLength + "," + breakLineY +
                        "M" + breakLineLength + "," + (breakLineY + breakLineSeparation) +
                        "C" + (breakLineLength * 2 / 3) + "," + ((breakLineY * 3 / 2) + breakLineSeparation) + "," + (breakLineLength / 3) + "," + ((-breakLineY / 2) + breakLineSeparation) + "," + 0 + "," + breakLineSeparation;
        var defs = figureContainer.append("defs");
        var breakFig = defs.append("g")
            .attr("id", "breakIcon");
        breakFig.append("path")
            .classed("maskIcon", true)
            .attr("transform", "rotate(90)")
            .attr("d", maskIcon);
        breakFig.append("path")
            .classed("breakIcon", true)
            .attr("transform", "rotate(90)")
            .attr("d", breakIcon);
        figureContainer.append("use")
            .attr("x", xScale(breakStart) + (breakLineSeparation * 3 / 2))
            .attr("y", (yScale(0) - (breakLineLength / 2)))
            .attr("xlink:href", "#breakIcon");
        figureContainer.append("use")
            .attr("x", xScale(breakStart) + (breakLineSeparation * 3 / 2))
            .attr("y", (yScale(1) - (breakLineLength / 2)))
            .attr("xlink:href", "#breakIcon");
    }
}