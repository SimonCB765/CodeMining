var figureWidth = 400;  // Height of the entire figure for an experiment's results including labels and title.
var figureHeight = 400;  // Width of the entire figure for an experiment's results including labels and title.
var figureMargin = {top: 20, right: 10, bottom: 20, left: 30};  // Margin around each individual figure.
var figureOffset = {top: 10, right: 10, bottom: 50, left: 60};  // The offset of the figures within the SVG element.
var svgWidth = (2 * (figureWidth + figureMargin.left + figureMargin.right)) + figureOffset.left + figureOffset.right;  // Width of the SVG element needed to hold both figures.
var svgHeight = figureHeight + figureMargin.top + figureMargin.bottom + figureOffset.top + figureOffset.bottom;  // Height of the SVG element needed to hold both figures.

// Create the SVG element.
var svg = d3.select("body")
    .append("svg")
        .attr("width", svgWidth)
        .attr("height", svgHeight);

// Add the axes labels.To rotate the label on the y axis we use a transformation. As this will alter the coordinate system being used
// to place the text, we perform the rotation around the point where the text is centered. This causes the text to rotate but not move.
var xAxisLabel = svg.append("text")
    .attr("class", "label")
    .attr("text-anchor", "middle")
    .attr("x", (svgWidth + figureOffset.left + figureOffset.right) / 2)
    .attr("y", svgHeight - (figureOffset.bottom / 2))
    .text("Variable Coefficients");
yAxisLabelLoc = { x : figureOffset.left * 2 / 3, y : svgHeight / 2 };
var yAxisLabel = svg.append("text")
    .attr("class", "label")
    .attr("text-anchor", "middle")
    .attr("x", yAxisLabelLoc.x)
    .attr("y", yAxisLabelLoc.y)
    .attr("transform", "rotate(-90, " + yAxisLabelLoc.x + ", " + yAxisLabelLoc.y + ")")
    .text("Coefficient Value Frequencies");

// Read the data in.
var dataAccessorFunction = function(d)
    {
        // If there is no entry on a line (because the variable was not used in the model), then return undefined so that it can be ignored later on.
        return {DisambiguationFirstModel : parseFloat(d.DisambiguationFirstModel), DisambiguationSecondModel : parseFloat(d.DisambiguationSecondModel),
                DiabetesFirstModel : parseFloat(d.DiabetesFirstModel), DiabetesSecondModel : parseFloat(d.DiabetesSecondModel)};
    }
d3.tsv("/Data/CoefComparisonResults.tsv", dataAccessorFunction, function(error, data)
    {
        // Create the figure the type 1 vs type 2 diabetes results.
        var data1v2 = data.filter(function(d)
            {
                // Only want rows where there is a coefficient value for the type 1 vs type 2 models. If there isn't a value for the first model,
                // then there won't be one for the second.
                return !isNaN(d.DisambiguationFirstModel);
            });
        data1v2 = data1v2.map(function(d, index) { return {First : d.DisambiguationFirstModel, Second : d.DisambiguationSecondModel}; })
        data1v2 = data1v2.filter(function(d) { return d.First !== 0 || d.Second !== 0; });
        var figure1v2 = svg.append("g")
            .attr("transform", "translate(" + (figureOffset.left + figureMargin.left) + ", " + (figureOffset.top + figureMargin.top) + ")");
        createFigure(figure1v2, data1v2, "Type 1 Vs. Type 2 Diabetes", 0.1);

        // Create the figure for the diabetes vs non-diabetes results.
        var dataDvND = data.filter(function(d)
            {
                // Only want rows where there is a coefficient value for the diabetes vs non-diabetes models. If there isn't a value for the first model,
                // then there won't be one for the second.
                return !isNaN(d.DiabetesFirstModel);
            });
        dataDvND = dataDvND.map(function(d, index) { return {First : d.DiabetesFirstModel, Second : d.DiabetesSecondModel}; })
        dataDvND = dataDvND.filter(function(d) { return d.First !== 0 || d.Second !== 0; });
        var figureDvND = svg.append("g")
            .attr("transform", "translate(" + (figureOffset.left + (2 * figureMargin.left) + figureWidth + figureMargin.right) + ", " + (figureOffset.top + figureMargin.top) + ")");
        createFigure(figureDvND, dataDvND, "Diabetes Vs. Non-diabetes", 0.02);
    }
);

function createFigure(figureContainer, dataArray, figureTitle, binWidth)
{
    // Define positioning variables.
    var spaceBetweenGraphs = 40;

    // Determine the maximum data value, and round it up to the nearest multiple of the bin width.
    var maxValue = d3.max(dataArray, function(d) { return Math.max(d.First, d.Second); });
    maxValue = Math.ceil(maxValue * (1 / binWidth)) / (1 / binWidth);

    // Determine he minimum data value, and round it down to the nearest multiple of the bin width.
    var minValue = d3.min(dataArray, function(d) { return Math.min(d.First, d.Second); });
    minValue = Math.floor(minValue * (1 / binWidth)) / (1 / binWidth);

    // Bin the data.
    var modelOneBinned = d3.layout.histogram()
        .value(function(d) { return d.First; })
        .bins((maxValue - minValue) / binWidth)
        .frequency(true)
        .range([minValue, maxValue])
        (dataArray);
    var maxCountFirstModel = d3.max(modelOneBinned, function(d) { return d.y; });
    var modelTwoBinned = d3.layout.histogram()
        .value(function(d) { return d.Second; })
        .bins((maxValue - minValue) / binWidth)
        .frequency(true)
        .range([minValue, maxValue])
        (dataArray);
    var maxCountSecondModel = d3.max(modelTwoBinned, function(d) { return d.y; });

    // Create the scales for the figures.
    var maxCount = Math.ceil(Math.max(maxCountFirstModel, maxCountSecondModel) * 0.1) / 0.1;  // Maximum count rounded up to nearest 10.
    var xScale = d3.scale.linear()
        .domain([minValue, maxValue])
        .range([0, figureWidth]);
    var yScaleTop = d3.scale.linear()
        .domain([0, maxCount])
        .range([(figureHeight - spaceBetweenGraphs) / 2, 0]);
    var yScaleBottom = d3.scale.linear()
        .domain([0, maxCount])
        .range([figureHeight, (figureHeight + spaceBetweenGraphs) / 2]);

    // Add the axes for the figure.
    var xAxisTop = d3.svg.axis()
        .scale(xScale)
        .orient("bottom");
    xAxisTop = figureContainer.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0, " + (figureHeight - spaceBetweenGraphs) / 2 + ")")
        .call(xAxisTop);
    var xAxisBottom = d3.svg.axis()
        .scale(xScale)
        .orient("bottom");
    xAxisBottom = figureContainer.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0, " + figureHeight + ")")
        .call(xAxisBottom);
    var yAxisTop = d3.svg.axis()
        .scale(yScaleTop)
        .orient("left");
    yAxisTop = figureContainer.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0, 0)")
        .call(yAxisTop);
    var yAxisBottom = d3.svg.axis()
        .scale(yScaleTop)
        .orient("left");
    yAxisBottom = figureContainer.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0, " + (figureHeight + spaceBetweenGraphs) / 2 + ")")
        .call(yAxisBottom);

    // Add the title to the figure.
    var title = figureContainer.append("text")
        .attr("class", "title")
        .attr("text-anchor", "middle")
        .attr("x", xScale(maxValue - ((maxValue - minValue) / 2)))
        .attr("y", 0)
        .text(figureTitle);

    // Add the histogram for the first model.
    var modelOneBars = figureContainer.selectAll(".modelOneBar")
        .data(modelOneBinned)
        .enter()
            .append("g")
                .attr("class", "bar")
                .attr("transform", function(d) { return "translate(" + xScale(d.x) + ", " + yScaleTop(d.y) + ")"; });
    modelOneBars.append("rect")
        .attr("x", 1)
        .attr("y", 0)
        .attr("width", function(d) { return xScale(maxValue) - xScale(maxValue - d.dx) - 1; })
        .attr("height", function(d) { return yScaleTop(0) - yScaleTop(d.y); });

    // Add the histogram for the second model.
    var modelTwoBars = figureContainer.selectAll(".modelTwoBar")
        .data(modelTwoBinned)
        .enter()
            .append("g")
                .attr("class", "bar")
                .attr("transform", function(d) { return "translate(" + xScale(d.x) + ", " + yScaleBottom(d.y) + ")"; });
    modelTwoBars.append("rect")
        .attr("x", 1)
        .attr("y", 0)
        .attr("width", function(d) { return xScale(maxValue) - xScale(maxValue - d.dx) - 1; })
        .attr("height", function(d) { return yScaleBottom(0) - yScaleBottom(d.y); });
}